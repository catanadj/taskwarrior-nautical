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
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import re
from decimal import Decimal, InvalidOperation
import shlex
import textwrap, shutil


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
# Locate nautical_core
# ------------------------------------------------------------------------------
HOOK_DIR = Path(__file__).resolve().parent
TW_DIR = HOOK_DIR.parent

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
    print(json.dumps(task), end="")


def _panel(
    title,
    rows,
    kind: str = "info",
    border_style: str | None = None,
    title_style: str | None = None,
    label_style: str | None = None,
):
    """
    Render a 2-column panel.

    Modes:
      - NAUTICAL_PANEL=rich (default): Rich bordered panel (color + compact).
      - NAUTICAL_PANEL=fast: plain aligned text (no Rich import; best for Termux).
      - NAUTICAL_PANEL=plain: legacy alias for "fast".
    """

    def _panel_fast(_title, _rows):
        """
        Fast plain renderer (NAUTICAL_PANEL=fast):
        - No Rich import.
        - Minimal ANSI colour to guide the eye (TTY-only, can be disabled via NO_COLOR=1 or NAUTICAL_FAST_COLOR=0).
        - Strips Rich markup tags (including bare "[/]" closers) while preserving bracketed literals like "[auto-due]" or "[123]".
        - Preserves row spacers (k is None).
        - Wraps to terminal width (capped to avoid overflow on narrow/mobile terminals).
        - Timeline is exploded into one entry per line (kept uncoloured).
        """
        def _strip_rich(s: str) -> str:
            if not s:
                return ""
            # Remove bare Rich reset tag.
            s = s.replace("[/]", "")
            # Remove Rich-style tags like [bold], [cyan], [/bold], [turquoise2], etc.
            # Keep bracketed literals that start with digits or contain '-' (e.g., [auto-due], [192]).
            s = re.sub(r"\[(\/?[A-Za-z_][A-Za-z0-9_ ]*)\]", "", s)
            return s

        # Colour control
        enable_color = bool(sys.stderr.isatty())
        if os.getenv("NO_COLOR") == "1":
            enable_color = False
        if (os.getenv("NAUTICAL_FAST_COLOR") or "").strip() == "0":
            enable_color = False

        ANSI = {
            "reset": "\x1b[0m",
            "bold": "\x1b[1m",
            "dim": "\x1b[2m",
            "cyan": "\x1b[36m",
            "green": "\x1b[32m",
            "red": "\x1b[31m",
            "yellow": "\x1b[33m",
        }

        def _c(txt: str, *styles: str) -> str:
            if not enable_color or not styles:
                return txt
            codes = "".join(ANSI[s] for s in styles if s in ANSI)
            return f"{codes}{txt}{ANSI['reset']}"

        # Terminal width (best-effort) + mobile-safe cap
        term_w = 0
        try:
            term_w = int(os.getenv("COLUMNS", "0") or 0)
        except Exception:
            term_w = 0
        if term_w <= 0:
            try:
                term_w = shutil.get_terminal_size(fallback=(100, 24)).columns
            except Exception:
                term_w = 100
        term_w = max(40, min(70, term_w))  # mobile-safe cap (prevents delimiter overflow)

        # Normalize rows and compute label width
        norm = []
        label_w = 0
        for k, v in _rows:
            if k is None:
                norm.append((None, None))
                continue
            ks = "" if k is None else _strip_rich(str(k)).strip()
            vs = "" if v is None else _strip_rich(str(v)).strip()
            norm.append((ks, vs))
            if ks.strip():
                label_w = max(label_w, len(ks))
        label_w = min(label_w, 22)

        # Timeline explode: split on "# N" boundaries.
        def _explode_timeline(t: str):
            t = (t or "").strip()
            if not t:
                return []
            hits = [m.start() for m in re.finditer(r"#\s*\d+", t)]
            if len(hits) <= 1:
                return [t]
            out = []
            for a, b in zip(hits, hits[1:] + [None]):
                seg = t[a:b].strip()
                if seg:
                    out.append(seg)
            return out or [t]

        def _wrap_chunks(text: str, width: int):
            return textwrap.wrap(
                text,
                width=width,
                break_long_words=False,
                break_on_hyphens=False,
            ) or [""]

        def _write_wrapped(prefix: str, text: str, value_style: str | None = None):
            # value_style is an ANSI-wrapped style key for the value part only
            avail = max(10, term_w - len(prefix))
            chunks = _wrap_chunks(text, avail)
            indent = " " * len(prefix)
            for i, ch in enumerate(chunks):
                pfx = prefix if i == 0 else indent
                if value_style:
                    sys.stderr.write(pfx + _c(ch, value_style) + "\n")
                else:
                    sys.stderr.write(pfx + ch + "\n")

        def _value_style_for(label: str, value: str):
            lk = (label or "").strip().lower()
            vlow = (value or "").lower()
            if lk in ("pattern",):
                return "cyan"
            if lk in ("natural", "basis", "root", "chain"):
                return "dim"
            if lk in ("next due", "first due"):
                if "overdue" in vlow:
                    return "red"
                if " in " in vlow or "Δ in" in vlow or "(in" in vlow:
                    return "green"
                return None
            # keep timeline uncoloured
            return None

        # Delimiters + title
        delim = "─" * term_w
        sys.stderr.write(_c(delim, "dim") + "\n")
        sys.stderr.write(_c(_strip_rich(str(_title)).rstrip(), "bold", "cyan") + "\n")
        sys.stderr.write(_c(delim, "dim") + "\n")

        # Rows
        for k, v in norm:
            if k is None:
                sys.stderr.write("\n")
                continue

            if not k:
                # continuation-only row
                _write_wrapped(" " * (label_w + 2), v, None)
                continue

            label_prefix = f"{k.ljust(label_w)}: "
            cont_prefix = " " * (label_w + 2)

            if k.lower() == "timeline":
                entries = _explode_timeline(v)
                if entries:
                    _write_wrapped(label_prefix, entries[0], None)
                    for ent in entries[1:]:
                        _write_wrapped(cont_prefix, ent, None)
                else:
                    sys.stderr.write(label_prefix + "\n")
                continue

            style = _value_style_for(k, v)
            _write_wrapped(label_prefix, v, style)

        sys.stderr.write(_c(delim, "dim") + "\n")

    panel_mode = (os.getenv("NAUTICAL_PANEL") or "").strip().lower()
    if panel_mode == "plain":
        panel_mode = "fast"
    if panel_mode == "fast" or os.getenv("NAUTICAL_PLAIN") == "1":
        return _panel_fast(title, rows)

    # ---- rich renderer ----
    try:
        if not sys.stderr.isatty():
            raise RuntimeError("no tty")
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
    except Exception:
        return _panel_fast(title, rows)

    THEMES = {
        "preview_anchor": {"border": "turquoise2", "title": "bright_cyan", "label": "light_sea_green"},
        "preview_cp": {"border": "deep_pink1", "title": "deep_pink1", "label": "deep_pink3"},
        "summary": {"border": "indian_red", "title": "indian_red", "label": "red"},
        "good": {"border": "green", "title": "green", "label": "green"},
        "warning": {"border": "yellow", "title": "yellow", "label": "yellow"},
        "error": {"border": "red", "title": "red", "label": "red"},
        "info": {"border": "blue", "title": "cyan", "label": "cyan"},
    }
    theme = THEMES.get(kind, THEMES["info"])

    border = border_style or theme["border"]
    tstyle = title_style or theme["title"]
    lstyle = label_style or theme["label"]

    console = Console(file=sys.stderr, force_terminal=True, highlight=False)
    t = Table.grid(padding=(0, 1), expand=False)
    t.add_column(style=f"bold {lstyle}", no_wrap=True, justify="right")
    t.add_column(style="white")

    for k, v in rows:
        if k is None:
            t.add_row("", "" if v is None else str(v))
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
            title=Text(str(title), style=f"bold {tstyle}"),
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
    next_keys = {"Next Due", "Link status", "Links left", "Limits"}
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
    next_keys = {"Next Due", "Link status", "Links left", "Limits"}
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
                ["task", "rc.hooks=off", "import", "-"],
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


def _root_uuid_from(task: dict) -> str:
    cur = task
    seen = set()
    lookups = 0
    u = task.get("uuid")
    prev = task.get("prevLink")
    while prev and prev not in seen and lookups < _MAX_UUID_LOOKUPS:
        seen.add(prev)
        lookups += 1
        obj = _export_uuid_short_cached(prev)
        if not obj:
            break
        u = obj.get("uuid") or u
        prev = obj.get("prevLink")
    return u

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

def tw_export_chain_or_fallback(seed_task: dict, env=None) -> list[dict]:
    """
    Fast path: export by chainID (new chains).
    Fallback: walk legacy chain via prevLink/nextLink starting from the root UUID.
    """
    # 1) Prefer chainID short token, if present
    short = (seed_task.get("chainID") or core.short_uuid(seed_task.get("uuid")) or "").strip()
    if short:
        try:
            out = _task(["rc.json.array=1", f"export chainID:{short}"], env=env)
            arr = json.loads(out) if out and out.strip().startswith("[") else []
            if arr:
                return arr
        except Exception:
            pass

    # 2) Fallback: legacy chain (no chainID) — BFS from root via prevLink/nextLink
    root_u = _root_uuid_from(seed_task) or seed_task.get("uuid")
    if not root_u:
        return [seed_task]

    seen, queue, bag = set(), [root_u], {}
    while queue and len(seen) < _MAX_CHAIN_WALK:
        u = queue.pop(0)
        if u in seen:
            continue
        seen.add(u)
        obj = _export_uuid_full(u, env=env)
        if not obj:
            continue
        bag[u] = obj
        for k in ("prevLink", "nextLink"):
            s = (obj.get(k) or "").strip()
            if s and s not in seen:
                queue.append(s)

    return list(bag.values()) or [seed_task]

# --- Chain analytics (cheap): chain age via one fast `_get` --------------------
@lru_cache(maxsize=256)
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
    print (age_days)
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
    # (Lint warnings can be added later.)

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
    """
    Return up to two *previous* tasks in chronological order (older first).
    """
    chain = []
    prev = current_task.get("prevLink")
    steps = 0
    lookups = 0
    while prev and len(chain) < 2 and steps < 6 and lookups < _MAX_UUID_LOOKUPS:
        obj = _export_uuid_short_cached(prev)
        if not obj:
            break
        chain.append(obj)
        prev = obj.get("prevLink")
        steps += 1
        lookups += 1
    # built newest→older; reverse to older→newer for display
    return list(reversed(chain))


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
        return "SKIP — Next anchor after completion time"
    if mode == "flex":
        return f"FLEX — Skip missed up to now; next after completion ({missed} missed since {due_s})"
    if basis == "missed":
        return f"ALL — Backfilling first of {missed} missed anchor(s) since {due_s}"
    if basis == "after_due":
        return "ALL (no missed) — Next anchor after original due"
    return "ALL — Next anchor after completion"


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
    expr_str = (parent.get("anchor") or "").strip()
    mode = (parent.get("anchor_mode") or "skip").strip().lower()
    if mode not in ("skip", "all", "flex"):
        raise ValueError(f"anchor_mode must be 'skip', 'all', or 'flex', got '{mode}'")

    expr_key = core.cache_key_for_task(expr_str, mode)
    if not expr_str:
        return (None, None, None)

    try:
        dnf = _validate_anchor_expr_cached(expr_str)
    except Exception as e:
        raise ValueError(f"Invalid anchor expression '{expr_str}': {str(e)}")

    end_dt, err = _safe_parse_datetime(parent.get("end"))
    if err:
        raise ValueError(f"end field: {err}")
    if not end_dt:
        return (None, None, None)

    end_d = _tolocal(end_dt).date()

    due_dt0, err = _safe_parse_datetime(parent.get("due"))
    if err:
        raise ValueError(f"due field: {err}")

    due_d_local = _tolocal(due_dt0).date() if due_dt0 else end_d
    default_seed = due_d_local
    seed_base = _root_uuid_from(parent) or "preview"


    info = {"mode": mode, "basis": None, "missed_count": 0, "missed_preview": []}

    if mode == "all":
        if end_d > due_d_local:
            missed = core.anchors_between_expr(
                dnf, due_d_local, end_d, default_seed, seed_base=seed_base
            )
            if missed:
                target = missed[0]
                info.update(
                    basis="missed",
                    missed_count=len(missed),
                    missed_preview=[x.isoformat() for x in missed[:5]],
                )
            else:
                target, _ = core.next_after_expr(
                    dnf, due_d_local, default_seed, seed_base=seed_base
                )
                info["basis"] = "after_due"
        else:
            target, _ = core.next_after_expr(
                dnf, due_d_local, default_seed, seed_base=seed_base
            )
            info["basis"] = "after_due"
    elif mode == "flex":
        target, _ = core.next_after_expr(dnf, end_d, default_seed, seed_base=seed_base)
        info["basis"] = "flex"
        if due_dt0 and end_d > due_d_local:
            missed = core.anchors_between_expr(
                dnf, due_d_local, end_d, default_seed, seed_base=seed_base
            )
            info.update(
                missed_count=len(missed),
                missed_preview=[x.isoformat() for x in missed[:5]],
            )
    else:
        target, _ = core.next_after_expr(dnf, end_d, default_seed, seed_base=seed_base)
        info["basis"] = "after_end"

    # Validate target date is not unreasonably far
    if not target:
        raise ValueError(
            "Could not compute next anchor date (pattern may not match future)"
        )

    # hh:mm strategy: pattern time > parent.due time > parent.end time
    hhmm = core.pick_hhmm_from_dnf_for_date(dnf, target, default_seed)
    if hhmm:
        hh, mm = hhmm
    elif due_dt0:
        dl = _tolocal(due_dt0)
        hh, mm = dl.hour, dl.minute
    else:
        el = _tolocal(end_dt)
        hh, mm = el.hour, el.minute

    due_local = core.build_local_datetime(target, (hh, mm))
    return due_local.astimezone(timezone.utc), info, dnf


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

    seed = _root_uuid_from(task) or "preview"
    nxt_local = _to_local_cached(next_due_utc)
    hhmm = (nxt_local.hour, nxt_local.minute)
    fut_local_date = nxt_local.date()

    fut_no = cur_no + 1
    while fut_no < cpmax:
        fut_no += 1
        fut_local_date, _ = core.next_after_expr(
            dnf, fut_local_date, default_seed=fut_local_date, seed_base=seed
        )

    fut_dt = core.build_local_datetime(fut_local_date, hhmm).astimezone(timezone.utc)
    return fut_dt


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
    tail = chain[-n:]
    base_no = core.coerce_int(tail[-1].get("link"), len(chain))
    labelw = max(4, len(f"#{base_no}"))
    lines = []
    start_no = base_no - (len(tail) - 1)
    for i, obj in enumerate(tail):
        no = start_no + i
        end = _dtparse(obj.get("end"))
        due = _dtparse(obj.get("due"))
        end_s = _fmtlocal(end) if end else "(no end)"
        delta = _fmt_on_time_delta(due, end)
        short = _short(obj.get("uuid"))
        lab = f"[bold]#{no:<{labelw-1}}[/]"
        line = f"{lab} {end_s} {delta} [dim]{short}[/]"
        if i == len(tail) - 1:
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


def _end_chain_summary(current: dict, reason: str, now_utc) -> None:
    kind_anchor = bool((current.get("anchor") or "").strip())
    kind = "anchor" if kind_anchor else "cp"

    # Prefer chainID export; fallback to legacy prev/next traversal
    chain = tw_export_chain_or_fallback(current)

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
            seed_base = _root_uuid_from(task) or "preview"
            base_hhmm = (_tolocal(child_due_utc).hour, _tolocal(child_due_utc).minute)
            fut_local_date = _tolocal(child_due_utc).date()
            
            for _ in range(allowed_future):
                if iterations >= _MAX_ITERATIONS:
                    break
                iterations += 1
                
                fut_no += 1
                try:
                    fut_local_date, _ = core.next_after_expr(
                        dnf,
                        fut_local_date,
                        default_seed=fut_local_date,
                        seed_base=seed_base,
                    )
                except Exception:
                    break
                
                fut_dt = core.build_local_datetime(
                    fut_local_date, base_hhmm
                ).astimezone(timezone.utc)
                
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
    print(json.dumps({"error": "Invalid anchor", "message": msg}))
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
    until = _dtparse(task.get("chainUntil"))
    if not until:
        return (None, None)

    cur_no = core.coerce_int(task.get("link"), 1)
    seed = _root_uuid_from(task) or "preview"

    nxt_local = _to_local_cached(next_due_utc)
    hhmm = (nxt_local.hour, nxt_local.minute)
    start_day = nxt_local.date() - timedelta(days=1)
    end_day = _to_local_cached(until).date()

    count = 0
    prev = start_day
    last_hit = None
    iterations = 0

    while iterations < _MAX_ITERATIONS:
        iterations += 1
        try:
            nxt_day, _ = core.next_after_expr(
                dnf, prev, default_seed=prev, seed_base=seed
            )
        except Exception:
            break

        if not nxt_day or nxt_day > end_day:
            break
        count += 1
        last_hit = nxt_day
        prev = nxt_day

    if count == 0 or not last_hit:
        return (None, None)

    final_no = cur_no + count
    final_dt = core.build_local_datetime(last_hit, hhmm).astimezone(timezone.utc)
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

def tw_export_chain(chain_id: str, since: datetime | None = None, extra: str | None = None) -> list[dict]:
    if not chain_id:
        return []
    args = ["task", "rc.hooks=off", "rc.json.array=on", "rc.verbose=nothing", f"chainID:{chain_id}"]
    if since:
        args.append(f"modified.after:{since.strftime('%Y-%m-%dT%H:%M:%S')}")
    if extra:
        args += shlex.split(extra)
    args.append("export")
    try:
        out = subprocess.check_output(args, text=True)
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
        print(json.dumps(new))
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
            print(json.dumps({"error": "Invalid CP", "message": str(e)}))
            sys.exit(1)
        except Exception as e:
            _panel("❌ CP Error", [("Validation", f"Unexpected error: {str(e)}")], kind="error")
            print(json.dumps({"error": "CP parsing error", "message": str(e)}))
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
        _end_chain_summary(new, f"Reached cap #{cap_no}", now_utc)
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

    try:
        child_short, stripped_attrs = _spawn_child(child)
    except Exception as e:
        _panel(
            "⛓ Chain error",
            [("Reason", f"Failed to spawn next link ({e})")],
            kind="error",
        )
        _print_task(new)
        return

    # Reflect link on parent for nice UX
    new["nextLink"] = child_short

    # Feedback panel
    fb = []
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


        child_obj = _export_uuid_short_cached(child_short)
        child_id = child_obj.get("id", "") if child_obj else ""
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

        child_obj = _export_uuid_short_cached(child_short)
        child_id = child_obj.get("id", "") if child_obj else ""
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
    main()

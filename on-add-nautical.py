#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
on-add-nautical.py (production)

Features:
- Weekly ranges require '..' (e.g., w:mon..fri).
- Auto-assign due when missing (anchor first, else cp).
- Panel logic:
    * If user set due -> First due = user's due; Next anchor shown separately.
    * If no due       -> First due = first anchor (or entry+cp) and due is auto-set.
- Optional warning if user's due isn't on an anchor day (constant toggle below).
- DST-safe datetime composition (local date+hh:mm -> UTC).
- Inclusive chainUntil behavior in preview.
"""

from __future__ import annotations

import sys, json, os, importlib, time, atexit
import re
from pathlib import Path
from datetime import timedelta, timezone, datetime
from functools import lru_cache
from contextlib import contextmanager
import shlex, subprocess
import pickle
import shutil
import textwrap
from collections import OrderedDict
try:
    import fcntl  # POSIX advisory lock
except Exception:
    fcntl = None


 # Ensure hook IO supports Unicode (emoji, symbols) in JSON output.
 # Python's json.dumps() defaults to ensure_ascii=True, which escapes non-ASCII
 # as "\\uXXXX". We prefer human-readable UTF-8 JSON for hook passthrough.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
try:
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ========= User-togglable constants =========================================
ANCHOR_WARN = True  # If True, warn when a user-provided due is not on an anchor day
UPCOMING_PREVIEW = 5  # How many future dates to preview.
_MAX_ITERATIONS = 2000
_MAX_PREVIEW_ITERATIONS = 750
_MAX_CHAIN_DURATION_YEARS = 5  # warn if chain extends this far
# ============================================================================

# --------------------------------------------------------------------------------------
# Locate and import nautical_core (looks in ~/.task/hooks, ~/.task, $TASKDATA, $TASKRC)
# --------------------------------------------------------------------------------------
HOOK_DIR = Path(__file__).resolve().parent
TW_DIR = HOOK_DIR.parent

_candidates: list[Path] = []

# --- Optional micro-profiler (stderr-only; enable with NAUTICAL_PROFILE=1 or 2)
_PROFILE_LEVEL = int(os.environ.get('NAUTICAL_PROFILE', '0') or '0')
_IMPORT_T0 = time.perf_counter()

def _add(p):
    if not p:
        return
    p = Path(p).expanduser().resolve()
    if p not in _candidates:
        _candidates.append(p)


_add(HOOK_DIR)  # dev copy next to hook
_add(TW_DIR)  # ~/.task
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

_IMPORT_MS = (time.perf_counter() - _IMPORT_T0) * 1000.0

class _Profiler:
    """Minimal, low-risk profiler for hook execution (stderr-only)."""

    def __init__(self, level: int = 0, import_ms: float | None = None):
        self.level = int(level or 0)
        self.enabled = self.level > 0
        self.import_ms = float(import_ms) if import_ms is not None else None
        self._t0 = time.perf_counter()
        self._events: list[tuple[str, float]] = []

    @contextmanager
    def section(self, name: str):
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._events.append((name, (time.perf_counter() - t0) * 1000.0))

    def add_ms(self, name: str, ms: float) -> None:
        if self.enabled:
            self._events.append((name, float(ms)))

    def emit(self) -> None:
        if not self.enabled:
            return
        total_ms = (time.perf_counter() - self._t0) * 1000.0
        lines = []
        lines.append(f"[NAUTICAL_PROFILE] total={total_ms:.1f}ms")
        if self.import_ms is not None:
            lines.append(f"  import_core={self.import_ms:.1f}ms")
        for name, ms in self._events:
            lines.append(f"  {name}={ms:.1f}ms")
        if self.level >= 2 and self._events:
            lines.append("  -- slowest --")
            for name, ms in sorted(self._events, key=lambda x: x[1], reverse=True)[:8]:
                lines.append(f"  {name}={ms:.1f}ms")
        sys.stderr.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
@lru_cache(maxsize=512)
def _parse_dt_any_cached(s: str):
    return core.parse_dt_any(s)


@lru_cache(maxsize=512)
def _fmt_dt_local_cached(dt):
    return core.fmt_dt_local(dt)


@lru_cache(maxsize=512)
def _to_local_cached(dt):
    return core.to_local(dt)


# ---- Optional cross-invocation disk cache for expensive anchor parsing (Termux-friendly) ----
# Default on; set NAUTICAL_DNF_DISK_CACHE=0 to disable.
_DNF_DISK_CACHE_ENABLED = (os.getenv("NAUTICAL_DNF_DISK_CACHE") or "1").strip().lower() in ("1", "true", "yes", "on")
_DNF_DISK_CACHE_PATH = HOOK_DIR / ".nautical_cache" / "dnf_cache.jsonl"
_DNF_DISK_CACHE_PATH_LEGACY = HOOK_DIR / ".nautical_cache" / "dnf_cache.pkl"
_DNF_DISK_CACHE = None  # OrderedDict[str, object]
_DNF_DISK_CACHE_DIRTY = False
_DNF_DISK_CACHE_MAX = 256
_DNF_DISK_CACHE_LOCK = _DNF_DISK_CACHE_PATH.with_suffix(".lock")
_DNF_DISK_CACHE_VERSION = 1
_DNF_DISK_CACHE_MAX_BYTES = 256 * 1024

def _core_sig() -> str:
    try:
        st = Path(core.__file__).stat()
        return f"{st.st_mtime_ns}:{st.st_size}"
    except Exception:
        return "unknown"

_CORE_SIG = _core_sig()

def _dnf_cache_key(expr: str) -> str:
    # tie cache entries to the current core build
    return f"{_CORE_SIG}|{expr}"

@contextmanager
def _dnf_cache_lock():
    """Best-effort lock for disk cache access."""
    if fcntl is not None:
        lf = None
        try:
            _DNF_DISK_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            lf = open(_DNF_DISK_CACHE_LOCK, "a", encoding="utf-8")
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        except Exception:
            lf = None
        try:
            yield
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
        return

    # Fallback: lockfile via O_EXCL (best-effort, short spin).
    fd = None
    for _ in range(6):
        try:
            _DNF_DISK_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            fd = os.open(str(_DNF_DISK_CACHE_LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            time.sleep(0.05)
        except Exception:
            break
    try:
        yield
    finally:
        try:
            if fd is not None:
                os.close(fd)
        except Exception:
            pass
        try:
            if fd is not None:
                os.unlink(_DNF_DISK_CACHE_LOCK)
        except Exception:
            pass

def _load_dnf_disk_cache() -> OrderedDict:
    global _DNF_DISK_CACHE, _DNF_DISK_CACHE_DIRTY
    if _DNF_DISK_CACHE is not None:
        return _DNF_DISK_CACHE
    _DNF_DISK_CACHE = OrderedDict()
    if not _DNF_DISK_CACHE_ENABLED:
        return _DNF_DISK_CACHE
    try:
        with _dnf_cache_lock():
            _DNF_DISK_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            if _DNF_DISK_CACHE_PATH.exists():
                try:
                    if _DNF_DISK_CACHE_PATH.stat().st_size > _DNF_DISK_CACHE_MAX_BYTES:
                        _diag(f"DNF cache too large; skipping load: {_DNF_DISK_CACHE_PATH}")
                        return _DNF_DISK_CACHE
                except Exception:
                    pass
                with open(_DNF_DISK_CACHE_PATH, "r", encoding="utf-8") as f:
                    parsed_any = False
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        parsed_any = True
                        if isinstance(obj, dict) and "version" in obj:
                            continue
                        key = None
                        val = None
                        if isinstance(obj, dict):
                            if "key" in obj and "value" in obj:
                                key = obj.get("key")
                                val = obj.get("value")
                            elif "k" in obj and "v" in obj:
                                key = obj.get("k")
                                val = obj.get("v")
                        if key is not None:
                            _DNF_DISK_CACHE[str(key)] = val
                    if not parsed_any:
                        try:
                            ts = int(time.time())
                            bad = _DNF_DISK_CACHE_PATH.with_suffix(f".corrupt.{ts}.jsonl")
                            os.replace(_DNF_DISK_CACHE_PATH, bad)
                            _diag(f"DNF cache quarantined: {bad}")
                        except Exception:
                            pass
            elif _DNF_DISK_CACHE_PATH_LEGACY.exists():
                # Legacy pickle format: raw dict or versioned dict.
                with open(_DNF_DISK_CACHE_PATH_LEGACY, "rb") as f:
                    obj = pickle.load(f)
                if isinstance(obj, dict) and "data" in obj:
                    data = obj.get("data")
                    if isinstance(data, dict):
                        _DNF_DISK_CACHE = OrderedDict(data.items())
                elif isinstance(obj, dict):
                    _DNF_DISK_CACHE = OrderedDict(obj.items())
                if _DNF_DISK_CACHE:
                    _DNF_DISK_CACHE_DIRTY = True
    except Exception:
        _DNF_DISK_CACHE = OrderedDict()
    return _DNF_DISK_CACHE

def _save_dnf_disk_cache() -> None:
    global _DNF_DISK_CACHE_DIRTY
    if not (_DNF_DISK_CACHE_ENABLED and _DNF_DISK_CACHE_DIRTY and isinstance(_DNF_DISK_CACHE, OrderedDict)):
        return
    try:
        with _dnf_cache_lock():
            _DNF_DISK_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            # trim oldest
            while len(_DNF_DISK_CACHE) > _DNF_DISK_CACHE_MAX:
                _DNF_DISK_CACHE.popitem(last=False)
            tmp = _DNF_DISK_CACHE_PATH.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                header = {"version": _DNF_DISK_CACHE_VERSION}
                f.write(json.dumps(header, ensure_ascii=False, separators=(",", ":")) + "\n")
                for k, v in _DNF_DISK_CACHE.items():
                    rec = {"key": k, "value": v}
                    f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
            os.replace(tmp, _DNF_DISK_CACHE_PATH)
            try:
                if _DNF_DISK_CACHE_PATH_LEGACY.exists():
                    _DNF_DISK_CACHE_PATH_LEGACY.unlink()
            except Exception:
                pass
            _DNF_DISK_CACHE_DIRTY = False
    except Exception:
        # cache write failures must never affect hook correctness
        pass

if _DNF_DISK_CACHE_ENABLED:
    atexit.register(_save_dnf_disk_cache)

@lru_cache(maxsize=256)
def _validate_anchor_expr_cached(expr: str):
    """
    Validate + parse anchor expression to DNF.

    Always caches in-memory (per invocation). Optionally caches across invocations
    when NAUTICAL_DNF_DISK_CACHE=1.
    """
    global _DNF_DISK_CACHE_DIRTY
    if _DNF_DISK_CACHE_ENABLED:
        cache = _load_dnf_disk_cache()
        k = _dnf_cache_key(expr)
        if k in cache:
            cache.move_to_end(k)
            return cache[k]

    dnf = core.validate_anchor_expr_strict(expr)

    if _DNF_DISK_CACHE_ENABLED:
        cache = _load_dnf_disk_cache()
        k = _dnf_cache_key(expr)
        cache[k] = dnf
        cache.move_to_end(k)
        _DNF_DISK_CACHE_DIRTY = True

    return dnf


def _human_delta(a, b, use_months_days=True):
    return core.humanize_delta(a, b, use_months_days=use_months_days)


def _short(u):
    try:
        s = str(u)
        return s[:8] if s else "—"
    except Exception:
        return "—"


def _root_uuid_from(task: dict) -> str | None:
    u = task.get("uuid")
    return str(u) if u else None


def _strip_quotes(s: str) -> str:
    s = (s or "").strip()
    return s[1:-1] if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"') else s

def _strip_rich_markup(s: str) -> str:
    # Remove simple Rich markup tags like [bold] or [/].
    return re.sub(r"\[[^\]]*\]", "", s or "")

def _panel(title, rows, kind: str = "info"):
    """
    Render a 2-column panel.

    Default renderer uses Rich for coloured borders and compact layout.
    For performance-sensitive environments (e.g., Termux), set:
        panel_mode="fast"   -> plain, aligned text output (no Rich import)
    """
    panel_mode = (core.PANEL_MODE or "").strip().lower()
    if panel_mode == "plain":
        panel_mode = "fast"
    if panel_mode == "line":
        panel_mode = "rich"
    if panel_mode == "fast":
        # ---- fast renderer (no Rich) ----
        def _strip_rich(s: str) -> str:
            # Remove Rich markup tags like [bold], [/], [cyan], etc.,
            # but keep bracketed literals such as [auto-due].
            # NOTE: Rich uses a bare closing tag "[/]" to reset style.
            s = s.replace("[/]", "")
            return re.sub(r"\[/?[A-Za-z0-9_ ]+\]", "", s)

        def _ansi_enabled() -> bool:
            # ANSI colours are enabled by default in fast mode when stderr is a TTY.
            # Disable with NO_COLOR=1 or fast_color=false in config.
            if not sys.stderr.isatty():
                return False
            if os.getenv("NO_COLOR") is not None:
                return False
            return bool(core.FAST_COLOR)

        _ANSI = _ansi_enabled()
        _RST = "\x1b[0m" if _ANSI else ""
        _C_TITLE = "\x1b[1;36m" if _ANSI else ""
        _C_CYAN = "\x1b[36m" if _ANSI else ""
        _C_DIM = "\x1b[2m" if _ANSI else ""
        _C_GREEN = "\x1b[32m" if _ANSI else ""
        _C_RED = "\x1b[31m" if _ANSI else ""
        _C_YELLOW = "\x1b[33m" if _ANSI else ""
        _C_BOLD = "\x1b[1m" if _ANSI else ""

        clean_rows = []
        for k, v in rows:
            if k is None:
                clean_rows.append((None, None))
            else:
                clean_rows.append((str(k), _strip_rich(str(v))))

        # Fast renderer: aligned labels + wrapping to terminal width (no Rich).
        try:
            term_w = int(os.getenv("COLUMNS", "0") or 0)
        except Exception:
            term_w = 0
        if not term_w:
            try:
                term_w = shutil.get_terminal_size((110, 24)).columns
            except Exception:
                term_w = 110
        term_w = max(40, min(70, term_w))

        label_w = 0
        for k, _v in clean_rows:
            if k is None:
                continue
            label_w = max(label_w, len(k))
        label_w = min(label_w, 28)

        def _value_style(label: str, value: str) -> str:
            lk = (label or "").strip().lower()
            lv = (value or "").strip().lower()

            if lk == "pattern":
                return _C_CYAN
            if lk == "natural":
                return _C_DIM
            if lk in {"first due", "next due"}:
                if "overdue" in lv:
                    return _C_BOLD + _C_RED
                return _C_BOLD + _C_GREEN
            if lk.startswith("[") or "warning" in lk or "invalid" in lk or "error" in lk:
                return _C_YELLOW
            if lk == "upcoming":
                return _C_DIM
            if lk == "chain":
                if "enabled" in lv:
                    return _C_DIM + _C_GREEN
                if "disabled" in lv:
                    return _C_DIM + _C_RED
                return _C_DIM
            return ""

        def _emit_wrapped(prefix0: str, indent: str, text: str, style: str = "") -> None:
            s = "" if text is None else str(text)
            if not s:
                sys.stderr.write(prefix0 + "\n")
                return
            avail = max(16, term_w - len(prefix0))
            first_line = True
            for raw_line in s.splitlines() or [""]:
                parts = textwrap.wrap(
                    raw_line,
                    width=avail,
                    break_long_words=False,
                    break_on_hyphens=False,
                ) or [""]
                for i, p in enumerate(parts):
                    pref = prefix0 if first_line and i == 0 else indent
                    if style and _ANSI:
                        sys.stderr.write(pref + style + p + _RST + "\n")
                    else:
                        sys.stderr.write(pref + p + "\n")
                first_line = False

        # Panel delimiters (helps visually bracket the fast output)
        delim = "─" * term_w
        if _ANSI:
            sys.stderr.write(_C_DIM + delim + _RST + "\n")
            sys.stderr.write(_C_TITLE + _strip_rich(str(title)) + _RST + "\n")
        else:
            sys.stderr.write(delim + "\n")
            sys.stderr.write(_strip_rich(str(title)) + "\n")

        for k, v in clean_rows:
            if k is None:
                sys.stderr.write("\n")
                continue
            label = f"{k:<{label_w}}"
            prefix0 = f"{label}  "
            indent = " " * len(prefix0)
            style = _value_style(k, v)
            _emit_wrapped(prefix0, indent, v, style=style)

        # Bottom delimiter
        if _ANSI:
            sys.stderr.write(_C_DIM + delim + _RST + "\n")
        else:
            sys.stderr.write(delim + "\n")
        return


    # ---- Rich renderer (default) ----
    # Cache heavy Rich objects across calls in a single invocation.
    global _RICH_CACHE
    try:
        _RICH_CACHE
    except NameError:
        _RICH_CACHE = {"loaded": False}

    if not _RICH_CACHE.get("loaded"):
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text
        except Exception:
            # fallback if Rich is unavailable
            sys.stderr.write(f"{title}\n")
            for k, v in rows:
                if k is None:
                    sys.stderr.write("\n")
                else:
                    sys.stderr.write(f"{k}: {v}\n")
            return
        _RICH_CACHE.update(
            {
                "loaded": True,
                "Console": Console,
                "Panel": Panel,
                "Table": Table,
                "Text": Text,
                "console": Console(file=sys.stderr, force_terminal=True, highlight=False),
            }
        )

    Console = _RICH_CACHE["Console"]
    Panel = _RICH_CACHE["Panel"]
    Table = _RICH_CACHE["Table"]
    Text = _RICH_CACHE["Text"]
    console = _RICH_CACHE["console"]

    THEMES = {
        "preview_anchor": {
            "border": "light_sea_green",
            "title": "light_sea_green",
            "label": "cyan",
        },
        "preview_cp": {
            "border": "hot_pink",
            "title": "bright_green",
            "label": "green",
        },
        "error": {"border": "red", "title": "red", "label": "red"},
        "warning": {"border": "yellow", "title": "yellow", "label": "yellow"},
        "info": {"border": "white", "title": "white", "label": "white"},
    }
    theme = THEMES.get(kind, THEMES["info"])

    table = Table.grid(padding=(0, 1))
    table.expand = True
    table.add_column(justify="right", style=theme["label"], no_wrap=True)
    table.add_column(justify="left")

    for k, v in rows:
        if k is None:
            table.add_row("", "")
            continue
        label = Text(str(k))
        value = Text.from_markup(str(v), style="white")
        table.add_row(label, value)

    panel = Panel(
        table,
        title=Text(title, style=theme["title"]),
        border_style=theme["border"],
        padding=(0, 1),
    )
    console.print(panel)

def _format_anchor_rows(rows: list[tuple[str, str]]) -> list[tuple[str | None, str]]:
    """Compact layout for anchor preview .
    """
    # Upcoming numbering to match the *chain link* index:
    #   link #1 → First due
    #   link #2 → Next anchor (if present)
    #   link #3+→ Upcoming list
    has_next_anchor = any(k == "Next anchor" for k, _ in rows)
    upcoming_start = 3 if has_next_anchor else 2

    # Extract delta row; later we inline it into "First due"
    delta_text = None
    for k, v in rows:
        if k == "Delta":
            delta_text = v
            break

    config_keys = {"Pattern", "Natural"}
    schedule_keys = {"First due", "Next anchor", "Scheduled", "Wait", "[auto-due]", "Upcoming"}
    limits_keys = {"Limits", "Final (until)"}

    config: list[tuple[str, str]] = []
    schedule: list[tuple[str, str]] = []
    limits: list[tuple[str, str]] = []
    warnings: list[tuple[str, str]] = []
    rand: list[tuple[str, str]] = []
    chain: list[tuple[str, str]] = []
    others: list[tuple[str, str]] = []

    for k, v in rows:
        # Skip standalone Delta row; we embed it into "First due"
        if k == "Delta":
            continue

        if k == "First due" and delta_text:
            v = f"{v}  [dim](Δ {delta_text})[/]"

        lk = (str(k).lower() if k is not None else "")

        if k in config_keys:
            config.append((k, v))
        elif k in schedule_keys:
            if k == "Upcoming" and v and v != "[dim]–[/]":
                # Add correct link numbers: 3,4,... if Next anchor present; else 2,3,...
                lines = v.splitlines()
                new_lines = []
                idx = upcoming_start
                for line in lines:
                    new_lines.append(f"[dim]{idx:>2} ▸[/] {line}")
                    idx += 1
                v = "\n".join(new_lines)
            schedule.append((k, v))
        elif k in limits_keys:
            limits.append((k, v))
        elif lk.startswith("warning") or lk.startswith("note"):
            warnings.append((k, v))
        elif k == "Rand":
            rand.append((k, v))
        elif k == "Chain":
            chain.append((k, v))
        else:
            others.append((k, v))

    # Any unknown rows → CONFIG so they don't vanish
    config.extend(others)

    out: list[tuple[str | None, str]] = []

    def _add(group: list[tuple[str, str]]):
        nonlocal out
        if not group:
            return
        if out:
            out.append((None, ""))  # spacer line
        out.extend(group)

    _add(config)
    _add(schedule)
    _add(limits)
    _add(warnings)
    _add(rand)
    _add(chain)

    return out or rows


def _diag(msg: str) -> None:
    if os.environ.get("NAUTICAL_DIAG") == "1":
        try:
            sys.stderr.write(f"[nautical] {msg}\n")
        except Exception:
            pass


def _run_task(
    cmd: list[str],
    *,
    env: dict | None = None,
    input_text: str | None = None,
    timeout: float = 3.0,
    retries: int = 2,
    retry_delay: float = 0.15,
) -> tuple[bool, str, str]:
    env = env or os.environ.copy()
    last_out = ""
    last_err = ""
    attempts = max(1, int(retries))
    for attempt in range(1, attempts + 1):
        try:
            r = subprocess.run(
                cmd,
                input=input_text,
                text=True,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
            last_out = r.stdout or ""
            last_err = r.stderr or ""
            if r.returncode == 0:
                return True, last_out, last_err
            if attempt < retries:
                time.sleep(retry_delay * attempt)
                continue
            return False, last_out, last_err
        except subprocess.TimeoutExpired:
            last_err = "timeout"
            if attempt < retries:
                time.sleep(retry_delay * attempt)
                continue
            return False, last_out, last_err
        except Exception as e:
            last_err = str(e)
            if attempt < retries:
                time.sleep(retry_delay * attempt)
                continue
            return False, last_out, last_err
    return False, last_out, last_err



def _format_cp_rows(rows: list[tuple[str, str]]) -> list[tuple[str | None, str]]:
    """Compact layout for classic cp preview.
    """
    # For cp chains, First due is link #1, Upcoming are 2,3,...
    upcoming_start = 2

    # Extract delta row; later we inline it into "First due"
    delta_text = None
    for k, v in rows:
        if k == "Delta":
            delta_text = v
            break

    config_keys = {"Period"}
    schedule_keys = {"First due", "Scheduled", "Wait", "[auto-due]", "Upcoming"}
    limits_keys = {"Limits", "Final (max)", "Final (until)"}

    config: list[tuple[str, str]] = []
    schedule: list[tuple[str, str]] = []
    limits: list[tuple[str, str]] = []
    warnings: list[tuple[str, str]] = []
    chain: list[tuple[str, str]] = []
    others: list[tuple[str, str]] = []

    for k, v in rows:
        if k == "Delta":
            continue

        if k == "First due" and delta_text:
            v = f"{v}  [dim](Δ {delta_text})[/]"

        lk = (str(k).lower() if k is not None else "")

        if k in config_keys:
            config.append((k, v))
        elif k in schedule_keys:
            if k == "Upcoming" and v and v != "[dim]–[/]":
                lines = v.splitlines()
                new_lines = []
                idx = upcoming_start
                for line in lines:
                    new_lines.append(f"[dim]{idx:>2} ▸[/] {line}")
                    idx += 1
                v = "\n".join(new_lines)
            schedule.append((k, v))
        elif k in limits_keys:
            limits.append((k, v))
        elif lk.startswith("warning") or lk.startswith("note"):
            warnings.append((k, v))
        elif k == "Chain":
            chain.append((k, v))
        else:
            others.append((k, v))

    config.extend(others)

    out: list[tuple[str | None, str]] = []

    def _add(group: list[tuple[str, str]]):
        nonlocal out
        if not group:
            return
        if out:
            out.append((None, ""))  # spacer line
        out.extend(group)

    _add(config)
    _add(schedule)
    _add(limits)
    _add(warnings)
    _add(chain)

    return out or rows


def _fail_and_exit(title: str, msg: str) -> None:
    # Pretty panel -> stderr
    _panel(f"❌ {title}", [("Message", msg)], kind="error")
    # Minimal feedback -> stdout (what Task expects when a hook fails)
    print(json.dumps({"error": title, "message": msg}, ensure_ascii=False))
    sys.exit(1)


def _error_and_exit(msg_tuples):
    _panel("❌ Invalid Chain", msg_tuples, kind="error")
    err = "Invalid chain"
    msg_parts = []
    for k, v in msg_tuples or []:
        if k:
            err = str(k)
            if v:
                msg_parts.append(f"{k}: {v}")
            else:
                msg_parts.append(str(k))
        elif v:
            msg_parts.append(str(v))
    msg = "; ".join(msg_parts) if msg_parts else "Invalid chain configuration."
    print(json.dumps({"error": err, "message": msg}, ensure_ascii=False))
    sys.exit(1)



# Local ISO string back to Taskwarrior (lets default-due adjusters run)
def _fmt_local_for_task(dt_utc):
    dl = core.to_local(dt_utc)
    return dl.strftime("%Y-%m-%dT%H:%M:%S")


# ------------------------------------------------------------------------------
# wait/scheduled preview helpers (panel only; no mutation on add)
# ------------------------------------------------------------------------------
def _fmt_td_no_seconds(td: timedelta) -> str:
    """Format timedelta as ±Dd HHh:MMm (seconds omitted)."""
    if not isinstance(td, timedelta):
        return "—"
    total_sec = int(td.total_seconds())
    sign = "-" if total_sec < 0 else "+"
    total_min = abs(total_sec) // 60
    days, rem = divmod(total_min, 60 * 24)
    hours, minutes = divmod(rem, 60)
    return f"{sign}{days}d {hours:02d}h:{minutes:02d}m"


def _append_wait_sched_rows(rows: list, task: dict, due_utc: datetime, auto_due: bool) -> None:
    """
    Append Scheduled/Wait lines to panel using LOCAL display, with UTC deltas to due.
    Also emit warnings when order due > scheduled > wait is violated.
    """
    if not (isinstance(rows, list) and isinstance(task, dict) and isinstance(due_utc, datetime)):
        return

    def _p(field: str) -> datetime | None:
        v = task.get(field)
        if not v:
            return None
        try:
            return core.parse_dt_any(v)
        except Exception:
            return None

    sched_dt = _p("scheduled")
    wait_dt = _p("wait")

    if sched_dt:
        rows.append((
            "Scheduled",
            f"[white]{core.fmt_dt_local(sched_dt)}[/]  [dim](Δ {_fmt_td_no_seconds(sched_dt - due_utc)})[/]"
        ))
    if wait_dt:
        rows.append((
            "Wait",
            f"[white]{core.fmt_dt_local(wait_dt)}[/]  [dim](Δ {_fmt_td_no_seconds(wait_dt - due_utc)})[/]"
        ))

    issues: list[str] = []
    if sched_dt and sched_dt > due_utc:
        issues.append(f"scheduled is after due by {_fmt_td_no_seconds(sched_dt - due_utc)}")
    if wait_dt and wait_dt > due_utc:
        issues.append(f"wait is after due by {_fmt_td_no_seconds(wait_dt - due_utc)}")
    if wait_dt and sched_dt and wait_dt > sched_dt:
        issues.append(f"wait is after scheduled by {_fmt_td_no_seconds(wait_dt - sched_dt)}")

    if issues:
        rows.append((
            "Warning",
            "[yellow]Wait/Sched: expected order due > scheduled > wait. " + "; ".join(issues) + ".[/]"
        ))
        if auto_due:
            rows.append((
                "Note",
                "[dim]This can happen when due is auto-assigned; adjust scheduled/wait if undesired.[/]"
            ))


# Helper to validate chainUntil is in the future
def _validate_until_not_past(until_dt, now_utc) -> tuple[bool, str | None]:
    """Check if chainUntil is in the past. Returns (is_valid, error_msg)."""
    if not until_dt:
        return (True, None)

    grace = timedelta(minutes=1)
    if until_dt < (now_utc - grace):
        past_by = now_utc - until_dt
        past_s = core.humanize_delta(until_dt, now_utc, use_months_days=False)
        return (False, f"chainUntil is in the past (was {past_s} ago)")

    return (True, None)


# Helper to check if due is in the past (warning only)
def _check_due_in_past(due_dt, now_utc) -> tuple[bool, str | None]:
    """
    Check if user-provided due is in the past.
    Returns (is_past, warning_msg).
    This is a warning only - allows backlog tasks.
    """
    if not due_dt:
        return (False, None)

    grace = timedelta(minutes=1)
    if due_dt < (now_utc - grace):
        ago = now_utc - due_dt
        ago_s = core.humanize_delta(due_dt, now_utc, use_months_days=False)
        return (True, f"Due date is in the past ({ago_s} ago).")

    return (False, None)


# Helper to warn if chain extends unreasonably far
def _validate_chain_duration_reasonable(
    until_dt, now_utc, first_due, kind
) -> tuple[bool, str | None]:
    """
    Warn if chain extends too far into future.
    Returns (is_reasonable, warning_msg).
    """
    if not until_dt:
        return (True, None)

    span = until_dt - now_utc
    years = span.days / 365.25

    if years > _MAX_CHAIN_DURATION_YEARS:
        return (False, f"Chain extends {years:.1f} years into future.")

    return (True, None)


# Helper to validate cp/anchor not missing
def _validate_kind_not_conflicting(cp_str, anchor_str) -> tuple[bool, str | None]:
    """
    Check if both cp and anchor are set (invalid).
    Returns (is_valid, error_msg).
    """
    has_cp = bool((cp_str or "").strip())
    has_anchor = bool((anchor_str or "").strip())

    if has_cp and has_anchor:
        return (False, "Cannot set both 'cp' and 'anchor'. Choose one.")

    return (True, None)


# Helper to validate chainMax > 0
def _validate_cpmax_positive(cpmax) -> tuple[bool, str | None]:
    """Check if chainMax is valid (positive). Returns (is_valid, error_msg)."""
    if cpmax <= 0:
        return (False, "chainMax must be > 0")

    return (True, None)


# Helper to safely parse with context
def _safe_parse_datetime(s, field_name) -> tuple[datetime | None, str | None]:
    """
    Parse datetime with error context.
    Returns (datetime, error_msg).
    """
    if not s:
        return (None, None)

    try:
        dt = core.parse_dt_any(s)
        if dt is None:
            return (None, f"{field_name}: Unrecognized datetime format '{s}'")
        return (dt, None)
    except ValueError as e:
        return (None, f"{field_name}: {str(e)}")
    except Exception as e:
        return (None, f"{field_name}: Unexpected parsing error: {str(e)}")


def _validate_no_legacy_colon_ranges(expr: str) -> tuple[bool, str | None]:
    """
    Detect legacy weekly range syntax and reject it.
    V2 delimiter contract requires '..' for ranges (e.g., 'w:mon..fri').
    Returns (is_valid, error_msg).
    """
    if not expr:
        return (True, None)
    
    expr = expr.strip()
    
    # Check for colon-separated day patterns (legacy format)
    # Patterns like: mon:wed:fri, mon:wed, tue:thu
    day_names = {"mon", "tue", "wed", "thu", "fri", "sat", "sun"}
    
    # Split by spaces first to handle DNF terms
    terms = expr.split()
    for term in terms:
        # Remove optional parentheses for DNF
        clean_term = term.strip("()")
        
        # Check if this looks like a legacy range/day list
        if ":" in clean_term or "-" in clean_term:
            parts = re.split(r"[:\-]", clean_term)
            # Check if all parts look like day abbreviations
            if len(parts) >= 2 and all(p.lower() in day_names for p in parts):
                legacy_example = clean_term
                return (
                    False,
                    f"Legacy weekly range '{legacy_example}' is not supported. Use '..' (e.g., 'w:mon..fri').",
                )
    
    return (True, None)

def _safe_parse_duration(s, field_name) -> tuple[timedelta | None, str | None]:
    """
    Parse duration with error context.
    Returns (timedelta, error_msg).
    """
    if not s:
        return (None, None)

    try:
        td = core.parse_cp_duration(s)
        if td is None:
            return (
                None,
                f"{field_name}: Invalid duration format '{s}' (expected: 3d, 2w, 1h, etc.)",
            )
        return (td, None)
    except ValueError as e:
        return (None, f"{field_name}: {str(e)}")
    except Exception as e:
        return (None, f"{field_name}: Unexpected parsing error: {str(e)}")


def _validate_anchor_syntax_strict(expr) -> tuple[bool, str | None]:
    """
    Strictly validate an anchor. Accepts string or DNF.
    Returns (True, None) on success, (False, message) on failure.
    """
    try:
        core.validate_anchor_expr_strict(expr)
        return True, None
    except Exception as e:
        return False, str(e)


def _validate_anchor_mode(mode_str) -> tuple[str, str | None]:
    """
    Validate and normalize anchor_mode. Returns (normalized_mode, error_msg).
    """
    mode = (mode_str or "skip").strip().lower()

    if mode not in ("skip", "all", "flex"):
        return (
            "skip",
            f"anchor_mode must be 'skip', 'all', or 'flex' (got '{mode}'). Defaulting to 'skip'.",
        )

    return (mode, None)

def tw_export_chain(chain_id: str, since: datetime | None = None, extra: str | None = None) -> list[dict]:
    if not chain_id:
        return []
    args = ["task", "rc.hooks=off", "rc.json.array=on", "rc.verbose=nothing", f"chainID:{chain_id}"]
    if since:
        args.append(f"modified.after:{since.strftime('%Y-%m-%dT%H:%M:%S')}")
    if extra:
        args += shlex.split(extra)
    args.append("export")
    ok, out, err = _run_task(args, timeout=3.0, retries=2)
    if not ok:
        _diag(f"tw_export_chain failed (chainID={chain_id}): {err.strip()}")
        return []
    try:
        data = json.loads(out.strip() or "[]")
        return data if isinstance(data, list) else [data]
    except Exception as e:
        _diag(f"tw_export_chain JSON parse failed: {e}")
        return []
# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    prof = _Profiler(level=_PROFILE_LEVEL, import_ms=_IMPORT_MS)
    if prof.enabled:
        atexit.register(prof.emit)
    with prof.section('read:stdin'):
        raw = sys.stdin.read().strip()
    if not raw:
        _fail_and_exit("Invalid input", "on-add must receive a single JSON task")
    try:
        with prof.section('parse:json'):
            task = json.loads(raw)
    except Exception:
        _fail_and_exit("Invalid input", "on-add must receive a single JSON task")

    with prof.section('clock:now'):
        now_utc = core.now_utc()
        now_local = core.to_local(now_utc)

    def _fmt(dt):
        return core.fmt_dt_local(dt)

    def _dt(s):
        return core.parse_dt_any(s)

    def _tol(dt):
        return core.to_local(dt)

    def _build_local(d, hmm):
        return core.build_local_datetime(d, hmm).astimezone(timezone.utc)

    # ========== EDGE CASE 1: Both cp and anchor set ==========
    cp_str = (task.get("cp") or "").strip()
    anchor_str = (task.get("anchor") or "").strip()
    _t_conf = time.perf_counter()
    is_valid, err = _validate_kind_not_conflicting(cp_str, anchor_str)
    prof.add_ms('validate:cp_vs_anchor', (time.perf_counter() - _t_conf) * 1000.0)
    if not is_valid:
        _error_and_exit([("Invalid chain config", err)])

    # Normalize
    has_cp = bool(cp_str)
    has_anchor = bool(anchor_str)
    kind = "anchor" if has_anchor else ("cp" if has_cp else None)

    # Default enable chain if cp/anchor present (respect explicit 'off')
    ch = (task.get("chain") or "").strip().lower()
    if (has_cp or has_anchor) and (not ch or ch == "off"):
        task["chain"] = "on"
        ch = "on"
        # Transparent notice to the user about the auto-enable:
        # _panel("Enabled chain", [("Reason", "cp/anchor present on add → set chain:on")], style="yellow")

    # Default link index for new Nautical roots.
    # Users rarely set this manually; it's Nautical bookkeeping.
    # Only set when missing/invalid AND task isn't already linked into a chain.
    if has_cp or has_anchor:
        linked_already = bool((task.get("prevLink") or "").strip() or (task.get("nextLink") or "").strip())
        if not linked_already:
            link_no = core.coerce_int(task.get("link"), 0)
            if link_no <= 0:
                task["link"] = 1
        # Transparent notice to the user about the auto-enable:
        # _panel("Enabled chain", [("Reason", "cp/anchor present on add → set chain:on")], style="yellow")

    # If nothing chain-related, just pass through
    if not kind:
        _t_out = time.perf_counter()
        print(json.dumps(task, ensure_ascii=False), end="")
        prof.add_ms('stdout:emit', (time.perf_counter() - _t_out) * 1000.0)
        return

    # ========== EDGE CASE 2: chainMax validation ==========
    cpmax = core.coerce_int(task.get("chainMax"), 0)
    if cpmax:
        is_valid, err = _validate_cpmax_positive(cpmax)
        if not is_valid:
            _error_and_exit([("Invalid chainMax", err)])

    # ========== EDGE CASE 3: Parse and validate chainUntil ==========
    until_dt, err = _safe_parse_datetime(task.get("chainUntil"), "chainUntil")
    if err:
        _error_and_exit([("Invalid chainUntil", err)])

    if until_dt:
        is_valid, err = _validate_until_not_past(until_dt, now_utc)
        if not is_valid:
            _error_and_exit([("Invalid chainUntil", err)])

    # Due seed (may be overwritten by auto-due)
    user_provided_due = bool(task.get("due"))
    due_dt = None
    past_due_warning = None
    if user_provided_due:
        due_dt, err = _safe_parse_datetime(task.get("due"), "due")
        if err:
            _error_and_exit([("Invalid due", err)])

        # ========== EDGE CASE 4: User due in past (warning only) ==========
        is_past, warn_msg = _check_due_in_past(due_dt, now_utc)
        if is_past:
            past_due_warning = warn_msg

    if due_dt is None:
        due_dt = now_utc

    due_local = _tol(due_dt)
    due_day = due_local.date()
    due_hhmm = (due_local.hour, due_local.minute)

    rows = []

    # [CHAINID] Stamp short root id on new chains (anchor/cp present, no existing chainID)
    try:
        if (task.get("anchor") or task.get("cp")) and not (task.get("chainID") or "").strip():
            task["chainID"] = core.short_uuid(task.get("uuid"))
    except Exception:
        # Never block task creation on bookkeeping
        pass
    # ==================================================================================
    # ANCHOR PREVIEW
    # ==================================================================================
    if kind == "anchor":
        # ========== EDGE CASE 5: Invalid anchor expression ==========
        is_valid, err = _validate_anchor_syntax_strict(anchor_str)
        if not is_valid:
            _error_and_exit([("Invalid anchor", err)])

        anchor_mode = ((task.get("anchor_mode") or "").strip().upper() or "ALL")
        _t0 = time.perf_counter()
        if core.ENABLE_ANCHOR_CACHE:
            pkg = core.build_and_cache_hints(anchor_str, anchor_mode, default_due_dt=due_dt)
            natural = pkg.get("natural") or core.describe_anchor_expr(anchor_str, default_due_dt=due_dt)
            dnf = pkg.get("dnf")  
        else:
            natural = core.describe_anchor_expr(anchor_str, default_due_dt=task.get("due"))
            dnf = core.validate_anchor_expr_strict(anchor_str)

        # ========== EDGE CASE 6: Invalid anchor_mode ==========
        mode, warn_msg = _validate_anchor_mode(task.get("anchor_mode"))
        task["anchor_mode"] = mode
        if warn_msg:
            rows.append(("Warning", f"[yellow]{warn_msg}[/]"))

        # Safe validate
        try:
            dnf = _validate_anchor_expr_cached(anchor_str)
        except Exception as e:
            _fail_and_exit("Invalid anchor", f"anchor syntax error: {e}")
        prof.add_ms('anchor:dnf', (time.perf_counter() - _t0) * 1000.0)


        tag = {
            "skip": "[bold bright_cyan]SKIP[/]",
            "all": "[bold yellow]ALL[/]",
            "flex": "[bold magenta]FLEX[/]",
        }.get(mode, "[bold bright_cyan]SKIP[/]")
        rows.append(("Pattern", f"[white]{anchor_str}[/]  {tag}"))
        try:
            rows.append(("Natural", f"[white]{core.describe_anchor_dnf(dnf, task)}[/]"))
        except Exception:
            pass

        base_local_date = due_day if user_provided_due else now_local.date()
        seed_base = (task.get("chainID") or "").strip() or _root_uuid_from(task) or "preview"

        # Fix: use a stable default_seed for /N gating
        interval_seed = base_local_date  # or first anchor day if you prefer

        def step_once(prev_local_date):
            try:
                nxt_date, _ = core.next_after_expr(
                    dnf,
                    prev_local_date,
                    default_seed=interval_seed,  # <-- fixed seed for the whole chain
                    seed_base=seed_base,
                )
                return nxt_date
            except Exception:
                return None


        first_date_local = step_once(base_local_date - timedelta(days=1))

        # ========== EDGE CASE 7: Anchor pattern doesn't match (no future dates) ==========
        if not first_date_local:
            _error_and_exit(
                [
                    (
                        "anchor pattern",
                        "No matching anchor dates found. Pattern may be invalid or too restrictive.",
                    )
                ]
            )


        fallback_hhmm = (due_hhmm if user_provided_due else (9, 0))

        def _norm_t_mod_early(v):
            if v is None:
                return []
            if isinstance(v, tuple) and len(v) == 2:
                return [v]
            if isinstance(v, list):
                out = []
                for it in v:
                    if isinstance(it, tuple) and len(it) == 2:
                        out.append(it)
                    elif isinstance(it, list) and len(it) == 2:
                        out.append((int(it[0]), int(it[1])))
                return out
            if isinstance(v, str):
                out = []
                for part in [p.strip() for p in v.split(",") if p.strip()]:
                    if len(part) == 5 and part[2] == ":" and part[:2].isdigit() and part[3:].isdigit():
                        out.append((int(part[:2]), int(part[3:])))
                return out
            return []

        def _term_fires_on_date_early(term, d):
            try:
                nxt, _ = core.next_after_expr(
                    [term],
                    d - timedelta(days=1),
                    default_seed=interval_seed,
                    seed_base=seed_base,
                )
                return nxt == d
            except Exception:
                return False

        def _expr_fires_on_date_early(d):
            try:
                nxt, _ = core.next_after_expr(
                    dnf,
                    d - timedelta(days=1),
                    default_seed=interval_seed,
                    seed_base=seed_base,
                )
                return nxt == d
            except Exception:
                return False

        def _times_for_date_early(d):
            times = set()
            for term in dnf:
                if _term_fires_on_date_early(term, d):
                    for atom in term:
                        mods = atom.get("mods") or {}
                        for hhmm in _norm_t_mod_early(mods.get("t")):
                            times.add(hhmm)
            return sorted(times)

        def _pick_occurrence_local(ref_dt_local, inclusive: bool):
            d0 = ref_dt_local.date()

            # Same-day: if expression fires today, try to pick a slot on the same day.
            if _expr_fires_on_date_early(d0):
                tlist = _times_for_date_early(d0) or [fallback_hhmm]
                for hhmm in tlist:
                    cand = core.build_local_datetime(d0, hhmm)
                    if (cand >= ref_dt_local) if inclusive else (cand > ref_dt_local):
                        return cand

            # Next matching date (strictly after d0)
            try:
                nxt_d, _ = core.next_after_expr(
                    dnf,
                    d0,
                    default_seed=interval_seed,
                    seed_base=seed_base,
                )
            except Exception:
                return None
            tlist = _times_for_date_early(nxt_d) or [fallback_hhmm]
            return core.build_local_datetime(nxt_d, tlist[0])

        _t_first = time.perf_counter()
        if user_provided_due:
            due_local_dt = _to_local_cached(due_dt)
            first_due_local_dt = _pick_occurrence_local(due_local_dt, inclusive=False)
            if not first_due_local_dt:
                _error_and_exit([("anchor pattern", "No matching anchor occurrences found after the provided due.")])
        else:
            first_due_local_dt = _pick_occurrence_local(now_local, inclusive=True)
            if not first_due_local_dt:
                _error_and_exit([("anchor pattern", "No matching anchor occurrences found.")])
        prof.add_ms('anchor:first_occurrence', (time.perf_counter() - _t_first) * 1000.0)

        first_hhmm = (first_due_local_dt.hour, first_due_local_dt.minute)
        first_date_local = first_due_local_dt.date()
        first_due_utc = first_due_local_dt.astimezone(timezone.utc)
        if user_provided_due:
            display_first_due_utc = due_dt
            rows.append(
                ("First due", f"[bold bright_green]{_fmt(display_first_due_utc)}[/]")
            )
            rows.append(("Next anchor", f"[white]{_fmt(first_due_utc)}[/]"))
        else:
            display_first_due_utc = first_due_utc
            rows.append(
                ("First due", f"[bold bright_green]{_fmt(display_first_due_utc)}[/]")
            )
            task["due"] = _fmt_local_for_task(first_due_utc)
            rows.append(
                (
                    "[auto-due]",
                    "Due date was not explicitly set; assigned to first anchor match.",
                )
            )

        # Scheduled/Wait preview (relative to First due)
        _append_wait_sched_rows(rows, task, display_first_due_utc, auto_due=(not user_provided_due))

        # Show past due warning if applicable
        if past_due_warning:
            rows.append(("Warning", f"[yellow]{past_due_warning}[/]"))

        if user_provided_due and ANCHOR_WARN:
            due_local_date = _to_local_cached(due_dt).date()
            first_after_due_date = step_once(due_local_date - timedelta(days=1))
            is_on_anchor_day = first_after_due_date == due_local_date
            if not is_on_anchor_day:
                rows.append(
                    (
                        "Note",
                        "[italic yellow]Your due is not an anchor day; chain follows anchors."
                        " To align, set due to a matching anchor day or omit due to auto-assign.[/]",
                    )
                )

        # ========== EDGE CASE 8: Chain duration warning ==========
        if until_dt:
            is_reasonable, warn_msg = _validate_chain_duration_reasonable(
                until_dt, now_utc, first_due_utc, "anchor"
            )
            if not is_reasonable and warn_msg:
                rows.append(("Warning", f"[yellow]{warn_msg}[/]"))

        cpmax = core.coerce_int(task.get("chainMax"), 0)

        exact_until_count = None
        final_until_dt = None
        if until_dt:
            end_day = _to_local_cached(until_dt).date()
            count = 0
            prev = first_date_local - timedelta(days=1)
            last = None
            iterations = 0
            for _ in range(_MAX_PREVIEW_ITERATIONS):
                if iterations >= _MAX_ITERATIONS:
                    break
                iterations += 1

                nxt = step_once(prev)
                if not nxt or nxt > end_day:
                    break
                count += 1
                last = nxt
                prev = nxt
            exact_until_count = max(0, count - 1)
            if last:
                final_hhmm = (
                    core.pick_hhmm_from_dnf_for_date(dnf, last, first_date_local)
                    or first_hhmm
                )
                final_until_dt = core.build_local_datetime(last, final_hhmm).astimezone(
                    timezone.utc
                )

        allow_by_max = (cpmax - 1) if (cpmax and cpmax > 0) else 10**9
        allow_by_until = exact_until_count if exact_until_count is not None else 10**9

        # Lint for *hints only*; do not fail on linter
        _t_lint = time.perf_counter()
        _, warns = core.lint_anchor_expr(anchor_str)
        prof.add_ms('anchor:lint', (time.perf_counter() - _t_lint) * 1000.0)
        if warns:
            _panel("ℹ️  Lint", [("Hint", w) for w in warns], kind="note")

        # Validator is the single source of truth
        _t_val = time.perf_counter()
        core.validate_anchor_expr_strict(anchor_str)
        prof.add_ms('anchor:validate_strict', (time.perf_counter() - _t_val) * 1000.0)


        preview_limit = max(0, min(UPCOMING_PREVIEW, allow_by_max, allow_by_until))

        preview = []
        colors = ["bright_cyan", "cyan", "bright_blue", "blue", "bright_black"]

        fallback_hhmm = first_hhmm

        def _norm_t_mod(v):
            if v is None:
                return []
            if isinstance(v, tuple) and len(v) == 2:
                return [v]
            if isinstance(v, list):
                out = []
                for it in v:
                    if isinstance(it, tuple) and len(it) == 2:
                        out.append(it)
                    elif isinstance(it, list) and len(it) == 2:
                        out.append((int(it[0]), int(it[1])))
                return out
            if isinstance(v, str):
                out = []
                for part in [p.strip() for p in v.split(",") if p.strip()]:
                    if len(part) == 5 and part[2] == ":" and part[:2].isdigit() and part[3:].isdigit():
                        out.append((int(part[:2]), int(part[3:])))
                return out
            return []

        def _term_fires_on_date(term, d):
            try:
                nxt, _ = core.next_after_expr(
                    [term],
                    d - timedelta(days=1),
                    default_seed=interval_seed,
                    seed_base=seed_base,
                )
                return nxt == d
            except Exception:
                return False

        def _times_for_date(d):
            times = set()
            for term in dnf:
                if _term_fires_on_date(term, d):
                    for atom in term:
                        mods = atom.get("mods") or {}
                        for hhmm in _norm_t_mod(mods.get("t")):
                            times.add(hhmm)
            return sorted(times)

        def _next_occurrence_after_local_dt(after_dt_local):
            d0 = after_dt_local.date()

            # Same-day: if expression fires today, try the next time slot today.
            try:
                nxt_date, _ = core.next_after_expr(
                    dnf,
                    d0 - timedelta(days=1),
                    default_seed=interval_seed,
                    seed_base=seed_base,
                )
                if nxt_date == d0:
                    tlist = _times_for_date(d0) or [fallback_hhmm]
                    for hhmm in tlist:
                        cand = core.build_local_datetime(d0, hhmm)
                        if cand > after_dt_local:
                            return cand
            except Exception:
                pass

            # Next matching date (strictly after d0)
            nxt_d = step_once(d0)
            if not nxt_d:
                return None
            tlist = _times_for_date(nxt_d) or [fallback_hhmm]
            return core.build_local_datetime(nxt_d, tlist[0])

        _t_prev = time.perf_counter()
        cur_dt = first_due_local_dt
        for i in range(preview_limit):
            nxt_dt = _next_occurrence_after_local_dt(cur_dt)
            if not nxt_dt:
                break
            dt_utc = nxt_dt.astimezone(timezone.utc)
            if until_dt and dt_utc > until_dt:
                break
            color = colors[min(i, len(colors) - 1)]
            preview.append(f"[{color}]{nxt_dt.strftime('%a %Y-%m-%d %H:%M %Z')}[/{color}]")
            cur_dt = nxt_dt
        prof.add_ms('anchor:preview_occurrences', (time.perf_counter() - _t_prev) * 1000.0)
        rows.append(("Upcoming", "\n".join(preview) if preview else "[dim]–[/]"))
        rows.append(
            (
                "Delta",
                f"[bright_yellow]{_human_delta(now_utc, display_first_due_utc, core.expr_has_m_or_y(dnf))}[/]",
            )
        )

        lim_parts = []
        if cpmax and cpmax > 0:
            lim_parts.append(f"max [bold yellow]{cpmax}[/]")
        if until_dt:
            lim_parts.append(f"until [bold yellow]{_fmt(until_dt)}[/]")
            if exact_until_count is not None:
                lim_parts.append(f"[white]{exact_until_count} more[/]")
        if lim_parts:
            rows.append(("Limits", " [dim]|[/] ".join(lim_parts)))
        if final_until_dt:
            rows.append(
                (
                    "Final (until)",
                    f"[bright_magenta]{_fmt(final_until_dt)}[/]  [dim]({_human_delta(now_utc, final_until_dt, True)})[/]",
                )
            )

        if "rand" in anchor_str.lower():
            base = _short(_root_uuid_from(task))
            rows.append(
                (
                    "Rand",
                    f"[dim italic]Preview uses provisional seed; final picks are chain-bound to {base}[/]",
                )
            )

        rows.append(
            (
                "Chain",
                "[bold green]enabled[/]" if ch == "on" else "[bold red]disabled[/]",
            )
        )
        rows = _format_anchor_rows(rows)
        _t_panel = time.perf_counter()
        _panel(
            "⚓︎ Anchor Preview",
            rows,
            kind="preview_anchor",
        )
        prof.add_ms('render:anchor_panel', (time.perf_counter() - _t_panel) * 1000.0)

        _t_out = time.perf_counter()
        print(json.dumps(task, ensure_ascii=False), end="")
        prof.add_ms('stdout:emit', (time.perf_counter() - _t_out) * 1000.0)
        return


    # ==================================================================================
    # CLASSIC CP PREVIEW
    # ==================================================================================
    td, err = _safe_parse_duration(cp_str, "cp")
    if err:
        _error_and_exit([("Invalid cp", err)])
    if not td:
        _error_and_exit([("Invalid cp", f"Couldn't parse duration from '{cp_str}'")])

    # ========== EDGE CASE 9: Chain duration warning for cp ==========
    if until_dt:
        is_reasonable, warn_msg = _validate_chain_duration_reasonable(
            until_dt, now_utc, now_utc + td if not user_provided_due else due_dt, "cp"
        )
        if not is_reasonable and warn_msg:
            rows.append(("Warning", f"[yellow]{warn_msg}[/]"))

    secs = int(td.total_seconds())
    preserve = secs % 86400 == 0

    entry_dt = _dt(task.get("entry")) if task.get("entry") else now_utc

    def add_period(dt):
        if preserve:
            dl = core.to_local(dt)
            base = core.build_local_datetime(
                (dl + timedelta(days=int(secs // 86400))).date(), (dl.hour, dl.minute)
            )
            return base.astimezone(timezone.utc)
        else:
            return (dt + td).replace(microsecond=0)

    if not user_provided_due:
        due_dt = add_period(entry_dt)
        task["due"] = _fmt_local_for_task(due_dt)
        rows.append(("[auto-due]", "Due was not set explicitly; assigned to entry+cp."))
    else:
        due_dt = _dt(task.get("due"))

    rows.append(("Period", f"[bold white]{cp_str}[/]"))
    rows.append(("First due", f"[bold bright_green]{_fmt(due_dt)}[/]"))

    # Scheduled/Wait preview (relative to First due)
    _append_wait_sched_rows(rows, task, due_dt, auto_due=(not user_provided_due))

    cpmax = core.coerce_int(task.get("chainMax"), 0)

    exact_until_count = None
    final_until_dt = None
    if until_dt:
        count = 0
        probe = due_dt
        last = None
        iterations = 0
        for _ in range(_MAX_PREVIEW_ITERATIONS):
            if iterations >= _MAX_ITERATIONS:
                break
            iterations += 1

            if probe > until_dt:
                break
            last = probe
            count += 1
            probe = add_period(probe)
        if count > 0:
            exact_until_count = max(0, count - 1)
            final_until_dt = last

    allow_by_max = (cpmax - 1) if (cpmax and cpmax > 0) else 10**9
    allow_by_until = exact_until_count if exact_until_count is not None else 10**9
    limit = max(0, min(UPCOMING_PREVIEW, allow_by_max, allow_by_until))

    preview, nxt = [], due_dt
    colors = ["bright_cyan", "cyan", "bright_blue", "blue", "bright_black"]
    for i in range(limit):
        nxt = add_period(nxt)
        if until_dt and nxt > until_dt:
            break
        color = colors[min(i, len(colors) - 1)]
        preview.append(f"[{color}]{_fmt(nxt)}[/{color}]")
    rows.append(("Upcoming", "\n".join(preview) if preview else "[dim]–[/]"))

    rows.append(("Delta", f"[bright_yellow]{_human_delta(now_utc, due_dt, False)}[/]"))

    lim_parts = []
    if cpmax and cpmax > 0:
        lim_parts.append(f"max [bold yellow]{cpmax}[/]")
        fmax = due_dt
        steps = max(0, cpmax - 1)
        for _ in range(steps):
            fmax = add_period(fmax)
        rows.append(
            (
                "Final (max)",
                f"[bright_magenta]{_fmt(fmax)}[/]  [dim]({_human_delta(now_utc, fmax, True)})[/]",
            )
        )

    if until_dt:
        lim_parts.append(f"until [bold yellow]{_fmt(until_dt)}[/]")
        if exact_until_count is not None:
            lim_parts.append(f"[white]{exact_until_count} more[/]")
        if final_until_dt:
            rows.append(
                (
                    "Final (until)",
                    f"[bright_magenta]{_fmt(final_until_dt)}[/]  [dim]({_human_delta(now_utc, final_until_dt, True)})[/]",
                )
            )

    if lim_parts:
        rows.append(("Limits", " [dim]|[/] ".join(lim_parts)))

    rows.append(
        ("Chain", "[bold green]enabled[/]" if ch == "on" else "[bold red]disabled[/]")
    )
    rows = _format_cp_rows(rows)
    _panel(
        "⛓ Recurring Chain Preview",
        rows,
        kind="preview_cp",
    )
    print(json.dumps(task, ensure_ascii=False), end="")



# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()

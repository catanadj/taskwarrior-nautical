#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared core for Taskwarrior Nautical hooks.

"""
from __future__ import annotations
import os, re, sys
import copy
import math
from datetime import datetime, timedelta, timezone, date
from functools import lru_cache, wraps
from calendar import month_name
from datetime import date as _date
import json, zlib, base64, hashlib, tempfile, time, random, subprocess
import difflib
from contextlib import contextmanager
try:
    import fcntl  # POSIX advisory lock
except Exception:
    fcntl = None


# ==============================================================================
# TABLE OF CONTENTS (major sections)
# 1) Config & defaults
# 2) Anchor parsing (DNF/ACF helpers)
# 3) Anchor cache & locking
# 4) Hook utilities (diag, run_task)
# 5) Taskwarrior helpers (exports, parsing, chain ops)
# ==============================================================================


# ==============================================================================
# SECTION: Config & defaults
# ==============================================================================
# --- TOML loading helpers ---


try:
    import tomllib  # Python 3.11+
except Exception:
    try:
        import tomli as tomllib  # Python 3.10 and earlier (pip install tomli)
    except Exception:
        tomllib = None


# --- Defaults ---
_DEFAULTS = {
    "wrand_salt": "nautical|wrand|v1",  # change to reshuffle weekly-rand streams
    "tz": "Europe/Bucharest",           # reserved for future DST/zone features
    "holiday_region": "",               # reserved for future holiday features
}

# --- Config cache ---
_CONF_CACHE = None

# --- Core constants ---
_CACHE_LOCK_RETRIES = 6
_CACHE_LOCK_SLEEP_BASE = 0.05
_CACHE_LOCK_JITTER = 0.0
_CACHE_LOCK_STALE_AFTER = 300.0

def _read_toml(path: str) -> dict:
    # Fast path: missing file => no config here
    try:
        if not path or not os.path.exists(path):
            return {}
    except Exception:
        return {}

    env_path = os.environ.get("NAUTICAL_CONFIG") or ""
    env_abs = os.path.abspath(os.path.expanduser(env_path)) if env_path else ""
    is_env_path = bool(env_abs and path == env_abs)

    # File exists, but we cannot parse TOML (Python < 3.11 and no tomli)
    if tomllib is None:
        if is_env_path:
            raise RuntimeError(
                f"NAUTICAL_CONFIG is set but TOML parser is unavailable for {path}. "
                "Install tomli or upgrade to Python 3.11+."
            )
        _warn_missing_toml_parser(path)
        return {}

    try:
        with open(path, "rb") as f:
            return tomllib.load(f) or {}
    except Exception as e:
        if is_env_path:
            raise RuntimeError(f"NAUTICAL_CONFIG parse failed for {path}: {e}")
        # Either show full details in explicit diagnostic mode
        if os.environ.get("NAUTICAL_DIAG") == "1":
            print(f"[nautical] Failed to parse TOML: {path}: {e}", file=sys.stderr)
        else:
            _warn_toml_parse_error(path, e)
        return {}


def _config_paths() -> list[str]:
    env_path = os.environ.get("NAUTICAL_CONFIG")
    if env_path:
        ap = os.path.abspath(os.path.expanduser(env_path))
        if (not os.path.exists(ap)) or os.path.isdir(ap):
            _warn_env_config_missing(env_path)
        return [ap]

    def _dedup(paths: list[str]) -> list[str]:
        seen = set()
        out = []
        for p in paths:
            if not p:
                continue
            ap = os.path.abspath(os.path.expanduser(p))
            if ap in seen:
                continue
            seen.add(ap)
            out.append(ap)
        return out

    def _candidates_in_dir(d: str) -> list[str]:
        d = os.path.abspath(os.path.expanduser(d))
        return [
            os.path.join(d, "config-nautical.toml"),
            os.path.join(d, "nautical.toml"),
        ]

    paths: list[str] = []

    # TASKRC contexts
    trc = os.environ.get("TASKRC")
    if trc:
        trc_abs = os.path.abspath(os.path.expanduser(trc))
        trc_dir = os.path.dirname(trc_abs)

        # next to TASKRC
        paths.extend(_candidates_in_dir(trc_dir))

        # sibling ".task" next to TASKRC directory (portable layouts)
        paths.extend(_candidates_in_dir(os.path.join(trc_dir, ".task")))

        # if TASKRC directory itself is ".task"
        if os.path.basename(trc_dir) == ".task":
            paths.extend(_candidates_in_dir(trc_dir))

    # module-adjacent
    moddir = os.path.dirname(os.path.abspath(__file__))
    paths.extend(_candidates_in_dir(moddir))

    # XDG config (explicit, then default)
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        paths.extend(_candidates_in_dir(os.path.join(xdg, "nautical")))
    paths.extend(_candidates_in_dir(os.path.expanduser("~/.config/nautical")))

    # Taskwarrior-centric placement
    paths.extend(_candidates_in_dir(os.path.expanduser("~/.task")))

    out = _dedup(paths)

    if os.environ.get("NAUTICAL_DIAG") == "1":
        try:
            print("[nautical] Config search order:", file=sys.stderr)
            for p in out:
                print(f"  - {p}", file=sys.stderr)
        except Exception:
            pass

    return out

def _warn_env_config_missing(env_path: str) -> None:
    _warn_once_per_day_any(
        "config_missing",
        "[nautical] NAUTICAL_CONFIG path missing; using defaults.",
    )
    if os.environ.get("NAUTICAL_DIAG") != "1":
        return
    ap = os.path.abspath(os.path.expanduser(env_path))
    print(
        "[nautical] NAUTICAL_CONFIG is set but the file is missing or invalid; defaults will be used.\n"
        f"          NAUTICAL_CONFIG={env_path}\n"
        f"          Resolved path: {ap}\n"
        "          Fix: create the file at that path or update NAUTICAL_CONFIG.\n",
        file=sys.stderr,
    )


def _normalize_keys(d: dict) -> dict:
    # allow users to write keys in any case
    out = {}
    for k, v in (d or {}).items():
        kk = str(k).strip().lower()
        out[kk] = v
    return out

def _load_config() -> dict:
    cfg = dict(_DEFAULTS)
    chosen = None

    paths = _config_paths()
    for p in paths:
        data = _read_toml(p)
        if data:
            cfg.update(_normalize_keys(data))
            chosen = p
            break

    if os.environ.get("NAUTICAL_DIAG") == "1":
        try:
            if chosen:
                print(f"[nautical] Using config: {chosen}", file=sys.stderr)
            else:
                print("[nautical] No config file found; using defaults.", file=sys.stderr)
                print("[nautical] Search order:", file=sys.stderr)
                for p in paths:
                    print(f"  - {p}", file=sys.stderr)
        except Exception:
            pass

    # normalize values
    cfg["wrand_salt"]      = str(cfg.get("wrand_salt") or _DEFAULTS["wrand_salt"])
    cfg["tz"]              = str(cfg.get("tz") or _DEFAULTS["tz"])
    cfg["holiday_region"]  = str(cfg.get("holiday_region") or "")
    return cfg



def _nautical_cache_dir() -> str:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(base, "nautical")


def _warn_once_per_day(key: str, message: str) -> None:
    """Persist a tiny sentinel so we do not spam hook output."""
    if os.environ.get("NAUTICAL_DIAG") != "1":
        return
    try:
        d = _nautical_cache_dir()
        os.makedirs(d, exist_ok=True)
        stamp_path = os.path.join(d, f".diag_{key}.stamp")

        today = date.today().isoformat()
        if os.path.exists(stamp_path):
            try:
                with open(stamp_path, "r", encoding="utf-8") as f:
                    if f.read().strip() == today:
                        return
            except Exception:
                pass

        with open(stamp_path, "w", encoding="utf-8") as f:
            f.write(today)
        try:
            print(message, file=sys.stderr)
        except Exception:
            pass
    except Exception:
        pass


def _warn_once_per_day_any(key: str, message: str) -> None:
    """Persist a tiny sentinel so we do not spam hook output (always on)."""
    try:
        d = _nautical_cache_dir()
        os.makedirs(d, exist_ok=True)
        stamp_path = os.path.join(d, f".diag_{key}.stamp")

        today = date.today().isoformat()
        if os.path.exists(stamp_path):
            try:
                with open(stamp_path, "r", encoding="utf-8") as f:
                    if f.read().strip() == today:
                        return
            except Exception:
                pass

        with open(stamp_path, "w", encoding="utf-8") as f:
            f.write(today)
        if os.environ.get("NAUTICAL_DIAG") == "1":
            try:
                print(message, file=sys.stderr)
            except Exception:
                pass
    except Exception:
        pass


def _warn_rate_limited_any(key: str, message: str, min_interval_s: float = 3600.0) -> None:
    """Emit a diagnostic warning at most once per min_interval_s (always on)."""
    try:
        d = _nautical_cache_dir()
        os.makedirs(d, exist_ok=True)
        stamp_path = os.path.join(d, f".diag_{key}.stamp")
        now = time.time()
        last = None
        if os.path.exists(stamp_path):
            try:
                with open(stamp_path, "r", encoding="utf-8") as f:
                    raw = f.read().strip()
                last = float(raw) if raw else None
            except Exception:
                last = None
        if last is not None and (now - last) < float(min_interval_s or 0.0):
            return
        with open(stamp_path, "w", encoding="utf-8") as f:
            f.write(str(now))
        if os.environ.get("NAUTICAL_DIAG") == "1":
            try:
                print(message, file=sys.stderr)
            except Exception:
                pass
    except Exception:
        pass


def _emit_cache_metrics() -> None:
    """Emit lru_cache metrics when NAUTICAL_DIAG_METRICS=1."""
    if os.environ.get("NAUTICAL_DIAG_METRICS") != "1":
        return
    lines = []
    try:
        lines.append(f"normalize_acf: {_normalize_spec_for_acf_cached.cache_info()}")
    except Exception:
        pass
    try:
        lines.append(f"year_pair: {_year_pair_cached.cache_info()}")
    except Exception:
        pass
    try:
        lines.append(f"parse_y_token: {_parse_y_token_cached.cache_info()}")
    except Exception:
        pass
    try:
        lines.append(f"expand_monthly: {expand_monthly_cached.cache_info()}")
    except Exception:
        pass
    try:
        lines.append(f"expand_weekly: {expand_weekly_cached.cache_info()}")
    except Exception:
        pass
    if not lines:
        return
    msg = "[nautical-metrics] " + " | ".join(lines)
    _warn_once_per_day("cache_metrics", msg)


def _clear_all_caches() -> None:
    """Clear all LRU caches (for long-running contexts)."""
    try:
        _normalize_spec_for_acf_cached.cache_clear()
    except Exception:
        pass
    try:
        _year_pair_cached.cache_clear()
    except Exception:
        pass
    try:
        _parse_y_token_cached.cache_clear()
    except Exception:
        pass
    try:
        expand_monthly_cached.cache_clear()
    except Exception:
        pass
    try:
        expand_weekly_cached.cache_clear()
    except Exception:
        pass


# -------- UI helpers ----------------------------------------------------------
_RICH_TAG_RE = re.compile(r"\[/\]|\[/?[A-Za-z0-9_ ]+\]")


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


# ==============================================================================
# SECTION: Panels & formatting helpers
# ==============================================================================
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
) -> None:
    try:
        if not sys.stderr.isatty():
            raise RuntimeError("no tty")
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text
    except Exception:
        emit_line(line)
        return

    theme = (themes or {}).get(kind) or (themes or {}).get("info") or {}
    border = border_style or theme.get("border", "blue")
    tstyle = title_style or theme.get("title", "cyan")

    console = Console(file=sys.stderr, force_terminal=True)
    body = Text(line)
    console.print(
        Panel(
            body,
            title=Text(str(title), style=f"bold {tstyle}"),
            border_style=border,
            expand=False,
            padding=(0, 1),
        )
    )


def render_panel(
    title,
    rows,
    *,
    kind: str = "info",
    panel_mode: str = "rich",
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
        mode = str(panel_mode or "").strip().lower()
        if mode in {"plain"}:
            mode = "fast"
        if mode == "line" and not allow_line:
            mode = "rich"
        if mode == "line" and line_force_rich_kinds and kind in line_force_rich_kinds:
            mode = "rich"

        if mode == "line":
            line = panel_line_from_rows(title, rows)
            if line:
                panel_line(title, line, kind=kind, themes=themes)
            return

        if mode == "fast":
            width = term_width_stderr()
            use_color = fast_color_enabled(fast_color=fast_color)
            RESET = ansi("0")
            BOLD = ansi("1") if use_color else ""
            DIM = ansi("2") if use_color else ""
            CYAN = ansi("36") if use_color else ""
            GREEN = ansi("32") if use_color else ""
            RED = ansi("31") if use_color else ""
            YELLOW = ansi("33") if use_color else ""

            delim = "─" * width
            sys.stderr.write(delim + "\n")
            sys.stderr.write((BOLD + CYAN + strip_rich_markup(str(title)) + RESET) + "\n")

            keys = [str(k) for (k, _v) in rows if k is not None]
            label_w = 0
            for k in keys:
                if len(k) > label_w:
                    label_w = len(k)
            label_w = min(label_width_max, max(label_width_min, label_w))

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

                k = strip_rich_markup(str(k))
                v = "" if v is None else strip_rich_markup(str(v))

                if k.lower().startswith("timeline"):
                    prefix0 = f"{k:<{label_w}} "
                    lines = [ln for ln in v.splitlines() if ln.strip() != ""] if "\n" in v else ([v] if v else [])
                    if lines:
                        emit_wrapped(prefix0, lines[0], width, style=None)
                        for ln in lines[1:]:
                            emit_wrapped(" " * len(prefix0), ln, width, style=None)
                    else:
                        emit_wrapped(prefix0, "", width, style=None)
                    continue

                prefix = f"{k:<{label_w}} "
                style = _style_for_row(k, v)
                emit_wrapped(prefix, v, width, style=style)

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
            width = term_width_stderr()
            use_color = fast_color_enabled(force=False, fast_color=fast_color)
            RESET = ansi("0")
            BOLD = ansi("1") if use_color else ""
            DIM = ansi("2") if use_color else ""
            CYAN = ansi("36") if use_color else ""
            GREEN = ansi("32") if use_color else ""
            RED = ansi("31") if use_color else ""
            YELLOW = ansi("33") if use_color else ""

            delim = "─" * width
            sys.stderr.write(delim + "\n")
            sys.stderr.write((BOLD + CYAN + strip_rich_markup(str(title)) + RESET) + "\n")

            keys = [str(k) for (k, _v) in rows if k is not None]
            label_w = 0
            for k in keys:
                if len(k) > label_w:
                    label_w = len(k)
            label_w = min(label_width_max, max(label_width_min, label_w))

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

                k = strip_rich_markup(str(k))
                v = "" if v is None else strip_rich_markup(str(v))

                if k.lower().startswith("timeline"):
                    prefix0 = f"{k:<{label_w}} "
                    lines = [ln for ln in v.splitlines() if ln.strip() != ""] if "\n" in v else ([v] if v else [])
                    if lines:
                        emit_wrapped(prefix0, lines[0], width, style=None)
                        for ln in lines[1:]:
                            emit_wrapped(" " * len(prefix0), ln, width, style=None)
                    else:
                        emit_wrapped(prefix0, "", width, style=None)
                    continue

                prefix = f"{k:<{label_w}} "
                style = _style_for_row(k, v)
                emit_wrapped(prefix, v, width, style=style)

            sys.stderr.write(delim + "\n")
            return

        theme = (themes or {}).get(kind) or (themes or {}).get("info") or {}
        border = theme.get("border", "blue")
        tstyle = theme.get("title", "cyan")
        lstyle = theme.get("label", "cyan")

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



def _warn_missing_toml_parser(config_path: str) -> None:
    pyver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    _warn_once_per_day_any(
        "missing_toml_parser_min",
        "[nautical] Config present but TOML parser unavailable; using defaults.",
    )
    msg = (
        "[nautical] Config detected but not loaded: TOML parser unavailable.\n"
        f"          Path: {config_path}\n"
        f"          Python: {pyver}\n"
        "          Fix: upgrade to Python 3.11+ (tomllib built-in) OR install tomli:\n"
        "               python3 -m pip install --user tomli\n"
        "          Tip: set NAUTICAL_DIAG=1 to print config search paths.\n"
    )
    _warn_once_per_day("missing_toml_parser", msg)


def _warn_toml_parse_error(config_path: str, err: Exception) -> None:
    _warn_once_per_day_any(
        "toml_parse_error_min",
        "[nautical] Config parse failed; using defaults.",
    )
    msg = (
        "[nautical] Config file found but could not be parsed; defaults will be used.\n"
        f"          Path: {config_path}\n"
        f"          Error: {err}\n"
        "          Fix: validate TOML syntax, or run with NAUTICAL_DIAG=1 for more context.\n"
    )
    _warn_once_per_day("toml_parse_error", msg)


def _get_config() -> dict:
    global _CONF_CACHE
    if _CONF_CACHE is None:
        _CONF_CACHE = _load_config()
    return _CONF_CACHE

_CONF = _get_config()

def _conf_raw(key: str):
    return _CONF.get(key)

def _conf_str(key: str, default: str) -> str:
    v = _conf_raw(key)
    if v is None:
        return str(default)
    s = str(v).strip()
    return s if s else str(default)

def _conf_int(
    key: str,
    default: int,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    v = _conf_raw(key)
    try:
        out = int(str(v).strip())
    except Exception:
        out = int(default)
    if min_value is not None and out < min_value:
        out = int(min_value)
    if max_value is not None and out > max_value:
        out = int(max_value)
    return out

def _conf_bool(
    key: str,
    default: bool = False,
    true_values: set[str] | None = None,
    false_values: set[str] | None = None,
) -> bool:
    v = _conf_raw(key)
    if v is None:
        return bool(default)
    s = str(v).strip().lower()
    if not s:
        return bool(default)
    if true_values and s in true_values:
        return True
    if false_values and s in false_values:
        return False
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off", "none"):
        return False
    return bool(default)

def _trueish(v, default=False):
    if v is None:
        return default
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "on")

ANCHOR_YEAR_FMT = "MD"
WRAND_SALT      = _CONF["wrand_salt"]
LOCAL_TZ_NAME   = _CONF["tz"]
HOLIDAY_REGION  = _CONF["holiday_region"]

ENABLE_ANCHOR_CACHE = _conf_bool("enable_anchor_cache", False)
ANCHOR_CACHE_DIR_OVERRIDE = _conf_str("anchor_cache_dir", "")   # optional custom path

# TTL is optional; 0 = no TTL
ANCHOR_CACHE_TTL = _conf_int("anchor_cache_ttl", 0, min_value=0)

# --- Hook-level toggles (shared config) -------------------------------------
CHAIN_COLOR_PER_CHAIN = _conf_bool(
    "chain_color_per_chain",
    False,
    true_values={"chain", "per-chain", "per"},
)
SHOW_TIMELINE_GAPS = _conf_bool(
    "show_timeline_gaps",
    True,
    false_values={"0", "no", "false", "off", "none"},
)
SHOW_ANALYTICS = _conf_bool(
    "show_analytics",
    True,
    false_values={"0", "no", "false", "off", "none"},
)
ANALYTICS_STYLE = _conf_str("analytics_style", "clinical").lower()
if ANALYTICS_STYLE not in ("coach", "clinical"):
    ANALYTICS_STYLE = "clinical"
ANALYTICS_ONTIME_TOL_SECS = _conf_int("analytics_ontime_tol_secs", 4 * 60 * 60, min_value=0)
VERIFY_IMPORT = _conf_bool("verify_import", True)
DEBUG_WAIT_SCHED = _conf_bool(
    "debug_wait_sched",
    False,
    true_values={"1", "yes", "true", "on"},
)
CHECK_CHAIN_INTEGRITY = _conf_bool(
    "check_chain_integrity",
    False,
    true_values={"1", "yes", "true", "on"},
)
PANEL_MODE = _conf_str("panel_mode", "rich").lower()
FAST_COLOR = _conf_bool("fast_color", True)
SPAWN_QUEUE_MAX_BYTES = _conf_int("spawn_queue_max_bytes", 524288, min_value=0)
SPAWN_QUEUE_DRAIN_MAX_ITEMS = _conf_int("spawn_queue_drain_max_items", 200, min_value=0)
MAX_CHAIN_WALK = _conf_int("max_chain_walk", 500, min_value=1)
MAX_ANCHOR_ITER = _conf_int("max_anchor_iterations", 128, min_value=32, max_value=1024)
MAX_LINK_NUMBER = _conf_int("max_link_number", 10000, min_value=1)
SANITIZE_UDA = _conf_bool("sanitize_uda", False, true_values={"1", "yes", "true", "on"})
SANITIZE_UDA_MAX_LEN = _conf_int("sanitize_uda_max_len", 1024, min_value=64, max_value=4096)
MAX_JSON_BYTES = _conf_int("max_json_bytes", 10 * 1024 * 1024, min_value=1024, max_value=100 * 1024 * 1024)
_CACHE_TTL_SECS = _conf_int("cache_ttl_secs", 3600, min_value=0)

def _ttl_lru_cache(maxsize: int = 128, ttl: float | None = None):
    ttl_val = _CACHE_TTL_SECS if ttl is None else ttl
    def _decorator(fn):
        cached = lru_cache(maxsize=maxsize)(fn)
        last = {"t": time.time()}
        @wraps(fn)
        def _wrapper(*args, **kwargs):
            if ttl_val and (time.time() - last["t"] > ttl_val):
                cached.cache_clear()
                last["t"] = time.time()
            return cached(*args, **kwargs)
        _wrapper.cache_clear = cached.cache_clear
        _wrapper.cache_info = cached.cache_info
        return _wrapper
    return _decorator

# ==============================================================================
# SECTION: Taskwarrior helpers
# ==============================================================================
def short_uuid(u: str | None) -> str:
    """Taskwarrior-style short uuid (first 8 hex)."""
    if not u or not isinstance(u, str):
        return ""
    s = u.strip().lower()
    if not s:
        return ""
    return s.split("-")[0] if "-" in s else s[:8]

def should_stamp_chain_id(task: dict) -> bool:
    """We stamp a chainID when task becomes/starts a nautical chain."""
    if not isinstance(task, dict): return False
    has_anchor = bool((task.get("anchor") or "").strip())
    has_cp     = bool((task.get("cp") or "").strip())
    already    = bool((task.get("chainID") or "").strip())
    return (has_anchor or has_cp) and not already

# ==============================================================================
# SECTION: Time & timezone helpers
# ==============================================================================
try:
    from zoneinfo import ZoneInfo

    _LOCAL_TZ = ZoneInfo(LOCAL_TZ_NAME)
except Exception:
    _LOCAL_TZ = None

def now_utc():
    """Get current UTC time without microseconds."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def to_local(dt_utc: datetime) -> datetime:
    """Convert UTC datetime to local timezone."""
    dt_utc = _ensure_utc(dt_utc)
    return dt_utc.astimezone(_LOCAL_TZ) if _LOCAL_TZ else dt_utc


def fmt_dt_local(dt_utc: datetime) -> str:
    """Format UTC datetime as local time string."""
    d = to_local(dt_utc)
    return d.strftime("%a %Y-%m-%d %H:%M %Z")


def fmt_isoz(dt_utc: datetime) -> str:
    """Format UTC datetime as ISO 8601 with Zulu time."""
    return _ensure_utc(dt_utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ensure_utc(dt_utc: datetime) -> datetime:
    """Return a timezone-aware UTC datetime."""
    if dt_utc.tzinfo is None:
        return dt_utc.replace(tzinfo=timezone.utc)
    return dt_utc.astimezone(timezone.utc)


# --- Date/time config ---
DATE_FORMATS = ("%Y%m%dT%H%M%SZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d")
UNTIL_COUNT_CAP = 1000
INTERSECTION_GUARD_STEPS = 256
DEFAULT_DUE_HOUR = 11

# --- Weekday constants ---
_WEEKDAYS = {
    "mon": 0,
    "monday": 0,
    "tue": 1,
    "tuesday": 1,
    "wed": 2,
    "wednesday": 2,
    "thu": 3,
    "thursday": 3,
    "fri": 4,
    "friday": 4,
    "sat": 5,
    "saturday": 5,
    "sun": 6,
    "sunday": 6,
}

# Canonical (V2) examples used in error messages / hints.
_CANON_WEEKLY_RANGE_EX = "w:mon..fri"
_CANON_WEEKLY_LIST_EX = "w:mon,wed,fri"
_MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

_MONTH_ALIAS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}



# Quarter mappings
_Q_FIRST_MONTH_RANGE = {  # full window for the quarter's first month
    1: "01-01..31-01",  # Jan
    2: "01-04..30-04",  # Apr
    3: "01-07..31-07",  # Jul
    4: "01-10..31-10",  # Oct
}
_Q_FIRST_DAY = {  # the first day of the quarter
    1: "01-01",  # Jan 1
    2: "01-04",  # Apr 1
    3: "01-07",  # Jul 1
    4: "01-10",  # Oct 1
}
_Q_LAST_DAY = {  # the last day of the quarter
    1: "31-03",  # Mar 31
    2: "30-06",  # Jun 30
    3: "30-09",  # Sep 30
    4: "31-12",  # Dec 31
}
_QUARTERS = {
    "q1": ((1, 1), (3, 31)),
    "q2": ((4, 1), (6, 30)),
    "q3": ((7, 1), (9, 30)),
    "q4": ((10, 1), (12, 31)),
}
_QUARTER_POS_MONTH = {
    1: {"s": 1, "m": 2, "e": 3},
    2: {"s": 4, "m": 5, "e": 6},
    3: {"s": 7, "m": 8, "e": 9},
    4: {"s": 10, "m": 11, "e": 12},
}


# Input/Output preference: "DM" (default) or "MD"
def _yearfmt():
    fmt = (globals().get("ANCHOR_YEAR_FMT") or "MD").upper()
    return "DM" if fmt == "DM" else "MD"



def _tok(d: int, m: int) -> str:
    return f"{d:02d}-{m:02d}" if _yearfmt() == "DM" else f"{m:02d}-{d:02d}"


def _tok_range(d1: int, m1: int, d2: int, m2: int) -> str:
    if _yearfmt() == "DM":
        # V2 delimiter contract: '..' denotes ranges.
        return f"{d1:02d}-{m1:02d}..{d2:02d}-{m2:02d}"
    else:
        return f"{m1:02d}-{d1:02d}..{m2:02d}-{d2:02d}"


# -------- Pre-compiled Regex Patterns ----------
_int_floatish_re = re.compile(r"^[+-]?\d+(?:\.0+)?$")
_cp_re = re.compile(
    r"^P(?:(?P<w>\d+)W)?(?:(?P<d>\d+)D)?(?:T(?:(?P<h>\d+)H)?(?:(?P<m>\d+)M)?(?:(?P<s>\d+)S)?)?$",
    re.I,
)
_hhmm_re = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")
_atom_head_re = re.compile(r"^(w|m|y)(?:/(\d+))?$")
_int_like_re = re.compile(r"^[+-]?\d+$")
_bd_re = re.compile(r"^(-?\d+)bd$")
_nth_weekday_re = re.compile(
    r"^(last|(?:-?\d+)(?:st|nd|rd|th)?)-?(mon|tue|wed|thu|fri|sat|sun)$"
)
_y_token_re = re.compile(r"^(\d{1,2})-([a-z]{3}|\d{1,2})$")
_next_prev_wd_re = re.compile(r"^(next|prev)-(mon|tue|wed|thu|fri|sat|sun)$")
_time_mod_re = re.compile(r"^t=(\d{2}:\d{2})$")
_day_offset_re = re.compile(r"^([+-]\d+)d$")
_nth_wd_re = re.compile(
    r"^(last|(?:-?\d+)(?:st|nd|rd|th)?)-?(mon|tue|wed|thu|fri|sat|sun)$"
)
_md_range_re = re.compile(r"(\d{2})-(\d{2})(?:\.\.(\d{2})-(\d{2}))?$")
_rand_mm_re = re.compile(r"^rand-(\d{2})$")
_year_range_colon_re = re.compile(r"^(\d{2})-(\d{2})\.\.(\d{2})-(\d{2})$")
_int_range_re = re.compile(r"^-?\d+\s*\.\.\s*-?\d+$")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

def _safe_match(pattern: re.Pattern, text: str, max_len: int = 256):
    """Defensive regex match to avoid pathological backtracking."""
    if text is None:
        return None
    if len(text) > max_len:
        raise ParseError("Expression too complex")
    return pattern.match(text)


def sanitize_text(v: str, max_len: int = 1024) -> str:
    """Remove control chars and clamp length for UDA safety."""
    if not isinstance(v, str):
        return v
    s = _CONTROL_CHARS_RE.sub("", v)
    if max_len > 0 and len(s) > max_len:
        if os.environ.get("NAUTICAL_DIAG") == "1":
            try:
                print(f"[nautical] UDA value truncated from {len(s)} to {max_len} chars", file=sys.stderr)
            except Exception:
                pass
        s = s[:max_len]
    return s


def sanitize_task_strings(task: dict, max_len: int = 1024) -> None:
    """In-place sanitize of string values in a task payload."""
    if not isinstance(task, dict):
        return
    for k, v in list(task.items()):
        if isinstance(v, str):
            cleaned = sanitize_text(v, max_len=max_len)
            if cleaned != v and os.environ.get("NAUTICAL_DIAG") == "1":
                try:
                    print(f"[nautical] UDA field truncated: {k}", file=sys.stderr)
                except Exception:
                    pass
            task[k] = cleaned


def _split_csv_tokens(spec: str) -> list[str]:
    return [t.strip() for t in str(spec or "").split(",") if t.strip()]


def _split_csv_lower(spec: str) -> list[str]:
    return [t.lower() for t in _split_csv_tokens(spec)]


# --- Interval bucket ---
def _iso_week_index(d: date) -> int:
    iso = d.isocalendar()
    return iso.year * 53 + iso.week  # monotonic-ish across years


def _month_index(d: date) -> int:
    return d.year * 12 + d.month


def _year_index(d: date) -> int:
    return d.year


# --- full-month helpers ---
def _static_month_last_day(mm: int) -> int:
    # Use 29 for February so leap years are fully covered; clamp at expansion time.
    return {
        1: 31,
        2: 29,
        3: 31,
        4: 30,
        5: 31,
        6: 30,
        7: 31,
        8: 31,
        9: 30,
        10: 31,
        11: 30,
        12: 31,
    }[mm]


def _month_from_alias(tok: str) -> int | None:
    s = (tok or "").strip().lower()
    if s.isdigit() and len(s) == 2:
        mm = int(s)
        return mm if 1 <= mm <= 12 else None
    return _MONTH_ALIAS.get(s)


def _year_full_months_span_token(m1: int, m2: int) -> str:
    """Full span across months [m1..m2], respecting ANCHOR_YEAR_FMT.

    V2 delimiter contract: use '..' for ranges.
    """
    return _tok_range(1, int(m1), 31, int(m2))


def _rewrite_month_names_to_ranges(spec: str) -> str:
    if not spec:
        return spec

    out = []
    for raw in _split_csv_tokens(spec):
        s = raw.lower()
        if ".." in s:
            a, b = [x.strip() for x in s.split("..", 1)]
            if a in _MONTH_ALIAS and b in _MONTH_ALIAS:
                m1, m2 = _MONTH_ALIAS[a], _MONTH_ALIAS[b]
                if m1 <= m2:
                    out.append(_tok_range(1, m1, _static_month_last_day(m2), m2))
                else:
                    out.append(raw)  # let validator complain about decreasing ranges
                continue
        if s in _MONTH_ALIAS:
            mm = _MONTH_ALIAS[s]
            out.append(_tok_range(1, mm, _static_month_last_day(mm), mm))
            continue
        out.append(raw)

    seen, dedup = set(), []
    for t in out:
        if t not in seen:
            dedup.append(t)
            seen.add(t)
    return ",".join(dedup)


# --- helpers for nth-weekday monthly /N gating ---
def _parse_nth_wd_tokens(spec: str):
    """Return list of (k, wd) for pure nth-weekday spec, else None.
    k is int in [-5..-1] ∪ [1..5], or -1 for 'last'."""
    toks = _split_csv_lower(spec)
    out = []
    for tok in toks:
        m = _nth_weekday_re.match(tok)
        if not m:
            return None
        n_raw, wd_s = m.group(1), m.group(2)
        if n_raw == "last":
            k = -1
        else:
            n_txt = re.sub(r"(st|nd|rd|th)$", "", n_raw)
            k = int(n_txt)
            if k == 0 or abs(k) > 5:  # safety;
                return None
        out.append((k, _WEEKDAYS[wd_s]))
    return out


def _month_has_any_nth(y: int, m: int, pairs: list[tuple[int, int]]) -> bool:
    """Does month (y,m) have ANY of the requested nth-weekdays?"""
    last = month_len(y, m)

    def kth(n, wd):
        if n == 0:
            return None
        if n > 0:
            d = date(y, m, 1)
            off = (wd - d.weekday()) % 7
            d = d + timedelta(days=off + (n - 1) * 7)
            return d.day if d.month == m else None
        d = date(y, m, last)
        off = (d.weekday() - wd) % 7
        d = d - timedelta(days=off + (abs(n) - 1) * 7)
        return d.day if d.month == m else None

    for n, wd in pairs:
        if kth(n, wd):
            return True
    return False


def _advance_to_next_allowed_month(y: int, m: int, pairs) -> tuple[int, int]:
    """Next (including current) month that has an nth-weekday match."""
    yy, mm = y, m
    for _ in range(24):  # guard
        if _month_has_any_nth(yy, mm, pairs):
            return (yy, mm)
        mm = 1 if mm == 12 else mm + 1
        if mm == 1:
            yy += 1
    return (y, m)  # fallback

def _unwrap_quotes(s: str) -> str:
    """Trim one pair of wrapping quotes ('...' or "...") if present."""
    if not s:
        return s
    s = str(s).strip()
    if len(s) >= 2 and ((s[0] == s[-1] == "'") or (s[0] == s[-1] == '"')):
        return s[1:-1].strip()
    return s

def _year_full_month_range_token(mm: int) -> str:
    """
    Return a yearly range token that covers the entire month `mm`
    formatted according to ANCHOR_YEAR_FMT.
    MD: 'MM-01..MM-31' ; DM: '01-MM..31-MM' (31 will be clamped later)
    """
    mm = int(mm)
    return _tok_range(1, mm, 31, mm)

def _mon_to_int(tok: str) -> int | None:
    s = (tok or "").strip().lower()
    if not s:
        return None
    if s.isdigit() and len(s) == 2:
        mm = int(s)
        return mm if 1 <= mm <= 12 else None
    return _MONTH_ALIAS.get(s)


def _rewrite_year_month_aliases_in_context(dnf: list[list[dict]]) -> list[list[dict]]:
    """
    In-place normalize yearly specs that are pure month references into
    full-month numeric ranges that the yearly gate understands.
      - 'y:04'           -> 'y:MM-01..MM-31'
      - 'y:jan'          -> 'y:MM-01..MM-31'
      - 'y:jan..apr'     -> 'y:01-MM..MM-31' (DM variant analogous)
      - 'y:04..06'       -> 'y:MM-01..MM-31'
      - 'y:apr,aug,12'   -> list of full-month ranges
    Quarters/other names should be rewritten earlier already; only
    handle obvious “month-only” forms here.
    """
    for term in dnf:
        for atom in term:
            if (atom.get("typ") or atom.get("type") or "").lower() != "y":
                continue
            spec = (atom.get("spec") or atom.get("value") or "").strip().lower()
            if not spec:
                continue

            new_tokens: list[str] = []
            changed = False
            for tok in _split_csv_tokens(spec):

                # 'mon1..mon2' or 'MM1..MM2' → full-month range
                if ".." in tok and "-" not in tok:
                    left, right = [x.strip() for x in tok.split("..", 1)]
                    m1, m2 = _mon_to_int(left), _mon_to_int(right)
                    if m1 and m2:
                        # Build a single cross-month range token; downstream clamping handles month-end.
                        new_tokens.append(_tok_range(1, m1, 31, m2))
                        changed = True
                        continue  # handled

                # Single month token (name or two-digit) → full-month
                m_single = _mon_to_int(tok)
                if m_single:
                    new_tokens.append(_year_full_month_range_token(m_single))
                    changed = True
                    continue

                # Else leave it as-is (numeric 'DD-MM' or 'DD-MM..DD-MM', 'rand', 'rand-MM', etc.)
                new_tokens.append(tok)

            if changed:
                atom["spec"] = ",".join(new_tokens)

    return dnf

# --- helpers used by the monthly /N branch ---
def _month_doms_for_spec(spec: str, y: date, m: int) -> list[int]:
    # must return day-of-month integers that match 'spec' in (y,m)
    try:
        return sorted(expand_monthly_cached(spec, y.year if isinstance(y, date) else y, m))
    except Exception:
        return []

def _month_has_hit(spec: str, y: int, m: int) -> bool:
    return bool(_month_doms_for_spec(spec, date(y, m, 1), m))

def _first_hit_after_probe_in_month(spec: str, y: int, m: int, probe: date) -> date | None:
    doms = _month_doms_for_spec(spec, date(y, m, 1), m)
    for d in doms:
        try:
            dt = date(y, m, d)
            if dt > probe:
                return dt
        except ValueError:
            continue
    return None

def _next_valid_month_on_or_after(spec: str, y: int, m: int) -> tuple[int, int]:
    yy, mm = y, m
    for _ in range(480):  # upper bound for safety
        if _month_has_hit(spec, yy, mm):
            return yy, mm
        mm += 1
        if mm > 12:
            yy += 1
            mm = 1
    return y, m  # fallback (should never hit)

def _advance_k_valid_months(spec: str, y: int, m: int, k: int) -> tuple[int, int]:
    """Advance forward by k valid months (k>=0)."""
    yy, mm = y, m
    steps = max(k, 0)
    while steps >= 0:
        # hop to the next calendar month first
        mm += 1
        if mm > 12:
            yy += 1
            mm = 1
        yy, mm = _next_valid_month_on_or_after(spec, yy, mm)
        steps -= 1
    return yy, mm

# --- Anchor Canonical Form (ACF) ----------------------------------------------


# Constants - no runtime config dependency
ACF_COMPRESSED = True  # Always compress for best storage
ACF_CHECKSUM_LEN = 8   # 8 chars = 32 bits of entropy


# Weekday normalize, with rand / rand*
_WD_ABBR = ["mon","tue","wed","thu","fri","sat","sun"]
_WEEKLY_ALIAS = {
    "wk": "mon..fri",
    "we": "sat..sun",
    "wd": "mon..fri",
}
_MONTHLY_ALIAS = {
    "ld": "-1",
    "lbd": "-1bd",
}

def _expand_weekly_aliases(spec: str) -> str:
    spec = (spec or "").strip().lower()
    if not spec:
        return spec
    toks = _split_csv_tokens(spec)
    out = []
    for tok in toks:
        t = (tok or "").strip().lower()
        if t in _WEEKLY_ALIAS:
            out.append(_WEEKLY_ALIAS[t])
        else:
            out.append(t)
    return ",".join([t for t in out if t])


# ==============================================================================
# SECTION: Anchor parsing (DNF/ACF helpers)
# ==============================================================================
# --- Token normalization ---
def _expand_monthly_aliases(spec: str) -> str:
    spec = (spec or "").strip().lower()
    if not spec:
        return spec
    toks = _split_csv_tokens(spec)
    out = []
    for tok in toks:
        t = (tok or "").strip().lower()
        if t in _MONTHLY_ALIAS:
            out.append(_MONTHLY_ALIAS[t])
        else:
            out.append(t)
    return ",".join([t for t in out if t])
def _normalize_weekday(s: str) -> str | None:
    s = (s or "").strip().lower()
    if not s:
        return None
    if s in ("rand", "rand*"):
        return s
    if s in _WD_ABBR:
        return s
    # allow numeric 1..7 (Mon..Sun)
    try:
        n = int(s)
        if 1 <= n <= 7:
            return _WD_ABBR[n-1]
    except Exception:
        pass
    return None

# ------------------------------------------------------------------------------
# ACF (Anchor Canonical Form) helpers
# ------------------------------------------------------------------------------
def _atom_sort_key(x: dict) -> tuple:
    # Stable order by (type, interval, spec-json, mods-json)
    sj = json.dumps(x.get("s"), separators=(",", ":"), sort_keys=True)
    mj = json.dumps(x.get("m"), separators=(",", ":"), sort_keys=True)
    return (x.get("t",""), int(x.get("i",1) or 1), sj, mj)

# Unpack function your validators call
def _acf_unpack(packed: str) -> dict:
    raw = base64.b85decode(packed.encode("ascii"))
    return json.loads(zlib.decompress(raw).decode("utf-8"))

def build_acf(expr: str) -> str:
    """
    Build canonical form with integrity protection.
    Format: "sha256:base85(zlib(json))"
    """
    if not expr or not expr.strip():
        return ""
    
    try:
        dnf = parse_anchor_expr_to_dnf_cached(expr)
    except Exception:
        # Parse failed - return sentinel
        return "!PARSE_ERROR"
    
    # Build canonical structure
    terms = []
    for term in dnf:
        atoms = []
        for a in term:
            typ = (a.get("typ") or "").lower()
            ival = int(a.get("ival") or 1)
            spec = a.get("spec") or ""
            mods = a.get("mods") or {}
            
            # Normalize spec
            norm_spec = _normalize_spec_for_acf(typ, spec)
            if norm_spec is None:
                continue  # Skip invalid atom
                
            # Build atom
            atom_obj = {
                "t": typ,
                "s": norm_spec,
                "m": _mods_to_acf(mods)
            }
            if ival != 1:
                atom_obj["i"] = ival
                
            atoms.append(atom_obj)
        
        if atoms:
            atoms.sort(key=lambda x: _atom_sort_key(x))
            terms.append(atoms)
    
    if not terms:
        return ""
    
    # Sort terms
    terms.sort(key=lambda x: json.dumps(x, sort_keys=True))
    
    # Pack with compression
    structure = {"terms": terms}
    json_str = json.dumps(structure, separators=(",", ":"), sort_keys=True)
    
    # Always compress
    compressed = zlib.compress(json_str.encode(), level=9)
    packed = base64.b85encode(compressed).decode("ascii")
    
    # Add checksum
    checksum = hashlib.sha256(packed.encode()).hexdigest()[:ACF_CHECKSUM_LEN]
    
    return f"{checksum}:{packed}"

def _normalize_spec_for_acf_uncached(typ: str, spec: str):
    """Comprehensive spec normalization (uncached)."""
    spec = (spec or "").strip().lower()
    
    if typ == "w":
        spec = _expand_weekly_aliases(spec)
        tokens = []
        for token in _split_csv_tokens(spec):
            if not token:
                continue
            if ".." in token:
                start, end = token.split("..", 1)
                s1 = _normalize_weekday(start)
                s2 = _normalize_weekday(end)
                if s1 and s2:
                    # Canonicalize all weekly ranges to V2 '..'.
                    tokens.append(f"{s1}..{s2}")
            else:
                s = _normalize_weekday(token)
                if s:
                    tokens.append(s)
        if not tokens:
            return None
        # keep distinct; ranges before singles for stability
        ranges = sorted([t for t in tokens if ".." in t])
        singles = sorted([t for t in tokens if ".." not in t])
        return ",".join(ranges + singles)

    
    elif typ == "m":
        # Monthly: canonicalize range delimiter to V2 '..' for cache stability.
        spec = _expand_monthly_aliases(spec)
        toks = []
        for token in _split_csv_tokens(spec):
            if not token:
                continue
            if ".." in token:
                a, b = [x.strip() for x in token.split("..", 1)]
                if a and b:
                    toks.append(f"{a}..{b}")
                else:
                    toks.append(token)
            else:
                toks.append(token)
        if not toks:
            return None
        ranges = sorted([t for t in toks if ".." in t])
        singles = sorted([t for t in toks if ".." not in t])
        return ",".join(ranges + singles)
    
    elif typ == "y":
        out = []
        for token in _split_csv_tokens(spec):
            m = re.fullmatch(r"(\d{2})-(\d{2})(?:\.\.(\d{2})-(\d{2}))?$", token)
            if not m:
                # assume already rewritten; if not, keep as string (worst case)
                out.append(token)
                continue
            a, b = int(m.group(1)), int(m.group(2))
            d1, m1 = _year_pair(a, b)
            if m.group(3):
                c, d = int(m.group(3)), int(m.group(4))
                d2, m2 = _year_pair(c, d)
                out.append({"m": m1, "d": d1, "to": {"m": m2, "d": d2}})
            else:
                out.append({"m": m1, "d": d1})
        return out

    return None


@_ttl_lru_cache(maxsize=512)
def _normalize_spec_for_acf_cached(typ: str, spec: str, fmt: str):
    typ = (typ or "").strip().lower()[:1]
    if typ not in ("w", "m", "y"):
        return None
    spec = (spec or "").strip().lower()[:256]
    fmt = "DM" if (fmt or "").upper() == "DM" else "MD"
    return _normalize_spec_for_acf_uncached(typ, spec)


def _normalize_spec_for_acf(typ: str, spec: str):
    """Comprehensive spec normalization (cached)."""
    res = _normalize_spec_for_acf_cached((typ or "").lower(), spec or "", _yearfmt())
    if isinstance(res, (list, dict)):
        return copy.deepcopy(res)
    return res

def is_valid_acf(acf_str: str) -> bool:
    if not acf_str:
        return False
    parts = acf_str.split(":", 2)  # c8, checksum, payload
    if len(parts) == 3 and parts[0].startswith("c"):
        _, checksum, payload = parts
    else:
        # legacy "checksum:payload"
        if ":" not in acf_str:
            return False
        checksum, payload = acf_str.split(":", 1)

    if len(checksum) != ACF_CHECKSUM_LEN:
        return False
    if hashlib.sha256(payload.encode()).hexdigest()[:ACF_CHECKSUM_LEN] != checksum:
        return False
    try:
        obj = _acf_unpack(payload)
        return bool(obj and "terms" in obj)
    except Exception:
        return False



def acf_to_original_format(acf_str: str) -> str:
    """
    Convert ACF back to approximate original expression.
    Useful for debugging and migration.
    """
    if not is_valid_acf(acf_str):
        return ""
    parts = acf_str.split(":", 2)
    if len(parts) == 3 and parts[0].startswith("c"):
        packed = parts[2]          # cN:checksum:payload
    else:
        packed = acf_str.split(":", 1)[1]  # checksum:payload
    obj = _acf_unpack(packed)
    if not obj:
        return ""
    
    terms_str = []
    for term in obj.get("terms", []):
        atoms_str = []
        for atom in term:
            typ = atom["t"]
            spec = atom["s"]
            ival = atom.get("i", 1)
            mods = atom.get("m", {})
            
            # Convert spec back to string
            spec_str = _acf_spec_to_string(typ, spec)
            
            # Build atom string
            atom_str = f"{typ}"
            if ival != 1:
                atom_str += f"/{ival}"
            atom_str += f":{spec_str}"
            
            # Add modifiers
            if mods:
                mods_str = _acf_mods_to_string(mods)
                if mods_str:
                    atom_str += mods_str
            
            atoms_str.append(atom_str)
        
        terms_str.append("+".join(sorted(atoms_str)))
    
    return " | ".join(sorted(terms_str))


@_ttl_lru_cache(maxsize=512)
def _year_pair_cached(a: int, b: int, fmt: str) -> tuple[int, int]:
    """Interpret (a,b) according to ANCHOR_YEAR_FMT; return (day, month)."""
    return (b, a) if fmt == "MD" else (a, b)


def _year_pair(a: int, b: int) -> tuple[int, int]:
    return _year_pair_cached(a, b, _yearfmt())

def _mods_to_acf(mods: dict) -> dict:
    """Keep only active modifiers in a compact, stable shape for ACF."""
    out = {}
    if not mods:
        return out
    t = mods.get("t")
    if t:
        if isinstance(t, tuple):
            out["t"] = f"{t[0]:02d}:{t[1]:02d}"
        elif isinstance(t, str) and _hhmm_re.fullmatch(t):
            out["t"] = t
    roll = mods.get("roll")
    if roll in ("pbd", "nbd", "nw", "next-wd", "prev-wd"):
        out["roll"] = roll
    if mods.get("bd"):
        out["bd"] = True
    if isinstance(mods.get("wd"), int):
        out["wd"] = int(mods["wd"])
    off = int(mods.get("day_offset") or 0)
    if off:
        out["+d"] = off
    return out

def _acf_mods_to_string(m: dict) -> str:
    """Turn ACF mods back into @-mod strings (best-effort, stable order)."""
    parts = []
    if m.get("t"):
        parts.append(f"@t={m['t']}")
    roll = m.get("roll")
    if roll in ("pbd", "nbd", "nw"):
        parts.append(f"@{roll}")
    elif roll in ("next-wd", "prev-wd"):
        wd = m.get("wd")
        wd_s = _WD_ABBR[wd] if isinstance(wd, int) and 0 <= wd < 7 else None
        if wd_s:
            parts.append(f"@{roll.split('-')[0]}-{wd_s}")
    if m.get("bd"):
        parts.append("@bd")
    if isinstance(m.get("+d"), int) and m["+d"]:
        parts.append(f"@{m['+d']:+d}d")
    return "".join(parts)

def _acf_spec_to_string(typ: str, spec) -> str:
    """Inverse of ACF spec normalization, back to anchor text."""
    if typ == "y" and isinstance(spec, list):
        out = []
        for item in spec:
            if isinstance(item, dict) and "m" in item and "d" in item:
                d1, m1 = item["d"], item["m"]
                if "to" in item and item["to"]:
                    d2, m2 = item["to"]["d"], item["to"]["m"]
                    out.append(_tok_range(d1, m1, d2, m2))
                else:
                    out.append(_tok(d1, m1))
            else:
                out.append(str(item))
        return ",".join(out)
    return str(spec)


# ==============================================================================
# SECTION: Anchor cache & locking
# ==============================================================================
# --- Cache directory discovery & IO ---
_CACHE_DIR = None

def _cache_dir() -> str:
    global _CACHE_DIR
    if _CACHE_DIR is not None:
        return _CACHE_DIR

    candidates = []
    if ANCHOR_CACHE_DIR_OVERRIDE:
        candidates.append(os.path.expanduser(ANCHOR_CACHE_DIR_OVERRIDE))

    here = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(here, ".nautical-cache"))

    taskdata = os.environ.get("TASKDATA")
    if taskdata:
        candidates.append(os.path.join(os.path.expanduser(taskdata), ".nautical-cache"))

    candidates.append(_nautical_cache_dir())
    if os.environ.get("NAUTICAL_ALLOW_TMP_CACHE") == "1":
        candidates.append(os.path.join(tempfile.gettempdir(), "nautical-cache"))

    def _ensure_cache_dir(path: str) -> bool:
        try:
            os.makedirs(path, mode=0o700, exist_ok=True)
            if os.path.isdir(path):
                try:
                    os.chmod(path, 0o700)
                except Exception:
                    pass
            return os.path.isdir(path) and os.access(path, os.W_OK)
        except Exception:
            return False

    for p in candidates:
        if not p:
            continue
        try:
            if _ensure_cache_dir(p):
                _CACHE_DIR = p
                return p
        except Exception:
            continue

    if os.environ.get("NAUTICAL_DIAG") == "1":
        try:
            print("[nautical] Anchor cache disabled: no writable cache dir found.", file=sys.stderr)
        except Exception:
            pass
    _CACHE_DIR = ""
    return ""

def _cache_key(acf: str, anchor_mode: str) -> str:
    payload = "|".join([acf, anchor_mode or "", ANCHOR_YEAR_FMT, WRAND_SALT, LOCAL_TZ_NAME, HOLIDAY_REGION, "nautical-cache|v1"])
    return hashlib.sha256(payload.encode()).hexdigest()[:16]

def _cache_path(key: str) -> str:
    base = _cache_dir()
    if not base:
        return ""
    return os.path.join(base, f"{key}.jsonz")

def _cache_lock_path(key: str) -> str:
    base = _cache_dir()
    if not base:
        return ""
    return os.path.join(base, f".{key}.lock")

@contextmanager
def safe_lock(
    path: str | os.PathLike,
    *,
    retries: int = 6,
    sleep_base: float = 0.05,
    jitter: float = 0.0,
    mode: int = 0o600,
    mkdir: bool = True,
    stale_after: float | None = 60.0,
):
    """Best-effort lock helper with fcntl (non-blocking) or O_EXCL fallback."""
    path_str = str(path) if path else ""
    if not path_str:
        yield False
        return

    def _sleep_once():
        try:
            delay = float(sleep_base or 0.0)
        except Exception:
            delay = 0.0
        if jitter:
            try:
                delay += random.uniform(0.0, float(jitter))
            except Exception:
                pass
        if delay > 0:
            time.sleep(delay)

    def _ensure_parent():
        if not mkdir:
            return
        try:
            parent = os.path.dirname(path_str)
            if parent:
                os.makedirs(parent, exist_ok=True)
        except Exception:
            pass

    def _lock_age(path: str) -> float | None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                head = f.read(64)
            parts = head.strip().split()
            if len(parts) >= 2:
                return time.time() - float(parts[1])
        except Exception:
            pass
        try:
            st = os.stat(path)
            return time.time() - float(st.st_mtime)
        except Exception:
            return None

    def _lock_stale_pid(path: str) -> bool:
        try:
            with open(path, "r", encoding="utf-8") as f:
                head = f.read(64)
            parts = head.strip().split()
            pid_str = parts[0] if parts else ""
            pid = int(pid_str)
            if pid <= 0:
                return True
            if stale_after is not None and len(parts) >= 2:
                try:
                    age = time.time() - float(parts[1])
                    if age < float(stale_after):
                        return False
                except Exception:
                    pass
            try:
                os.kill(pid, 0)
                return False
            except PermissionError:
                return False
            except ProcessLookupError:
                return True
            except Exception:
                return False
        except Exception:
            return False

    tries = max(1, int(retries or 0))

    if fcntl is not None:
        lf = None
        acquired = False
        _ensure_parent()
        try:
            fd = os.open(path_str, os.O_CREAT | os.O_RDWR, mode)
            try:
                os.fchmod(fd, mode)
            except Exception:
                pass
            lf = os.fdopen(fd, "a", encoding="utf-8")
            for _ in range(tries):
                try:
                    fcntl.flock(lf.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except Exception:
                    _sleep_once()
        except Exception:
            lf = None
        try:
            yield acquired
        finally:
            try:
                if acquired and lf is not None:
                    fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                if lf is not None:
                    lf.close()
            except Exception:
                pass
        return

    fd = None
    acquired = False
    for _ in range(tries):
        _ensure_parent()
        try:
            fd = os.open(path_str, os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode)
            try:
                os.fchmod(fd, mode)
            except Exception:
                pass
            try:
                payload = f"{os.getpid()} {int(time.time())}\n"
                os.write(fd, payload.encode("ascii", "replace"))
            except Exception:
                pass
            acquired = True
            break
        except FileExistsError:
            pid_stale = _lock_stale_pid(path_str)
            age_stale = False
            if stale_after is not None:
                age = _lock_age(path_str)
                if age is not None and age >= float(stale_after):
                    age_stale = True
            if pid_stale and age_stale:
                try:
                    os.unlink(path_str)
                except Exception:
                    pass
            else:
                _sleep_once()
        except Exception:
            break
    try:
        yield acquired
    finally:
        try:
            if acquired and fd is not None:
                os.close(fd)
        except Exception:
            pass
        try:
            if acquired and fd is not None:
                os.unlink(path_str)
        except Exception:
            pass

@contextmanager
def _cache_lock(key: str):
    """Best-effort per-key lock for cache writes. Yields True if acquired."""
    lock_path = _cache_lock_path(key)
    if not lock_path:
        yield False
        return
    with safe_lock(
        lock_path,
        retries=_CACHE_LOCK_RETRIES,
        sleep_base=_CACHE_LOCK_SLEEP_BASE,
        jitter=_CACHE_LOCK_JITTER,
        mode=0o600,
        mkdir=True,
        stale_after=_CACHE_LOCK_STALE_AFTER,
    ) as acquired:
        yield acquired


# ==============================================================================
# SECTION: Hook utilities (diag, run_task)
# ==============================================================================
# --- Diagnostic logging ---
_DIAG_LOG_REDACT_KEYS = frozenset({"description", "annotation", "annotations", "note", "notes"})


def _redact_dict(data: dict, redact_keys: frozenset) -> dict:
    out = {}
    for k, v in (data or {}).items():
        if k in redact_keys:
            out[k] = "[redacted]"
        else:
            out[k] = v
    return out


def diag_log_redact(msg: str, redact_keys: frozenset | None = None) -> str:
    """Redact sensitive keys from JSON msg for diagnostic logs."""
    keys = redact_keys or _DIAG_LOG_REDACT_KEYS
    if isinstance(msg, dict):
        return _redact_dict(msg, keys)
    try:
        data = json.loads(msg)
        if isinstance(data, dict):
            for k in list(data.keys()):
                if k in keys:
                    data[k] = "[redacted]"
            return json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        pass
    return msg


def _diag_log_path(data_dir: str | None = None) -> str:
    base = data_dir or os.environ.get("TASKDATA")
    if base:
        return os.path.join(os.path.abspath(os.path.expanduser(base)), ".nautical_diag.jsonl")
    return os.path.join(os.path.expanduser("~/.task"), ".nautical_diag.jsonl")


def diag_log(msg: str, hook_name: str, data_dir: str | None = None) -> None:
    """Append a JSONL diagnostic log entry (when NAUTICAL_DIAG_LOG=1)."""
    if os.environ.get("NAUTICAL_DIAG_LOG") != "1":
        return
    path = _diag_log_path(data_dir)
    max_bytes = int(os.environ.get("NAUTICAL_DIAG_LOG_MAX_BYTES") or 262144)
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
    except Exception:
        pass
    try:
        if max_bytes > 0 and os.path.exists(path):
            try:
                st = os.stat(path)
                if st.st_size > max_bytes:
                    overflow = path.replace(".jsonl", f".overflow.{int(time.time())}.jsonl")
                    os.replace(path, overflow)
            except Exception:
                pass
        fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
        try:
            os.fchmod(fd, 0o600)
        except Exception:
            pass
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "hook": hook_name,
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "cwd": os.getcwd(),
        }
        if data_dir:
            payload["data_dir"] = str(data_dir)
        if isinstance(msg, dict):
            red = diag_log_redact(msg)
            if isinstance(red, dict):
                payload["msg"] = str(red.get("msg") or red.get("message") or "")
                payload["data"] = red
            else:
                payload["msg"] = str(red)
        else:
            payload["msg"] = diag_log_redact(str(msg))
        with os.fdopen(fd, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    except Exception:
        pass


def diag(msg, hook_name: str = "nautical", data_dir: str | None = None) -> None:
    """Write diagnostics to stderr when NAUTICAL_DIAG=1 and append to diag log when NAUTICAL_DIAG_LOG=1."""
    if os.environ.get("NAUTICAL_DIAG") == "1":
        try:
            sys.stderr.write(f"[nautical] {msg}\n")
        except Exception:
            pass
    diag_log(msg, hook_name, data_dir)

# --- Subprocess runner ---
def run_task(
    cmd: list[str],
    *,
    env: dict | None = None,
    input_text: str | None = None,
    timeout: float = 3.0,
    retries: int = 2,
    retry_delay: float = 0.15,
    use_tempfiles: bool = False,
) -> tuple[bool, str, str]:
    """Run a subprocess; returns (ok, stdout, stderr). Uses env or os.environ.copy()."""
    env = env or os.environ.copy()
    last_out = ""
    last_err = ""
    attempts = max(1, int(retries))
    for attempt in range(1, attempts + 1):
        out_path = err_path = None
        out_f = err_f = None
        try:
            if use_tempfiles:
                try:
                    out_f = tempfile.NamedTemporaryFile(delete=False)
                    err_f = tempfile.NamedTemporaryFile(delete=False)
                    out_path = out_f.name
                    err_path = err_f.name
                except Exception:
                    out_f = err_f = None
                    out_path = err_path = None
            text_mode = not bool(out_f)
            if not text_mode and isinstance(input_text, str):
                input_text = input_text.encode("utf-8")
            elif text_mode and isinstance(input_text, (bytes, bytearray)):
                input_text = input_text.decode("utf-8", "replace")
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=(out_f if out_f is not None else subprocess.PIPE),
                stderr=(err_f if err_f is not None else subprocess.PIPE),
                text=text_mode,
                encoding=("utf-8" if text_mode else None),
                errors=("replace" if text_mode else None),
                close_fds=True,
                env=env,
            )
            try:
                out, err = proc.communicate(input=input_text, timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    out, err = proc.communicate(timeout=1.0)
                except Exception:
                    out, err = "", ""
                try:
                    if out_f is not None:
                        out_f.close()
                    if err_f is not None:
                        err_f.close()
                except Exception:
                    pass
                if out_path:
                    try:
                        with open(out_path, "rb") as f:
                            out = f.read().decode("utf-8", "replace")
                    except Exception:
                        out = ""
                    try:
                        os.unlink(out_path)
                    except Exception:
                        pass
                if err_path:
                    try:
                        with open(err_path, "rb") as f:
                            err = f.read().decode("utf-8", "replace")
                    except Exception:
                        err = ""
                    try:
                        os.unlink(err_path)
                    except Exception:
                        pass
                last_err = "timeout"
                if attempt < retries:
                    delay = retry_delay * (2 ** (attempt - 1))
                    jitter = random.uniform(0.0, retry_delay) if retry_delay > 0 else 0.0
                    time.sleep(delay + jitter)
                    continue
                return False, out or "", last_err
            try:
                if out_f is not None:
                    out_f.close()
                if err_f is not None:
                    err_f.close()
            except Exception:
                pass
            if out_path:
                try:
                    with open(out_path, "rb") as f:
                        out = f.read().decode("utf-8", "replace")
                except Exception:
                    out = ""
                try:
                    os.unlink(out_path)
                except Exception:
                    pass
            if err_path:
                try:
                    with open(err_path, "rb") as f:
                        err = f.read().decode("utf-8", "replace")
                except Exception:
                    err = ""
                try:
                    os.unlink(err_path)
                except Exception:
                    pass
            last_out = out or ""
            last_err = err or ""
            if proc.returncode == 0:
                return True, last_out, last_err
            if attempt < retries:
                delay = retry_delay * (2 ** (attempt - 1))
                jitter = random.uniform(0.0, retry_delay) if retry_delay > 0 else 0.0
                time.sleep(delay + jitter)
                continue
            return False, last_out, last_err
        except Exception as e:
            last_err = str(e)
            try:
                if out_path and os.path.exists(out_path):
                    os.unlink(out_path)
                if err_path and os.path.exists(err_path):
                    os.unlink(err_path)
            except Exception:
                pass
            if attempt < retries:
                delay = retry_delay * (2 ** (attempt - 1))
                jitter = random.uniform(0.0, retry_delay) if retry_delay > 0 else 0.0
                time.sleep(delay + jitter)
                continue
            return False, last_out, last_err
    return False, last_out, last_err


def is_lock_error(err: str) -> bool:
    """Check if stderr indicates a Taskwarrior/database lock error."""
    e = (err or "").lower()
    return (
        "database is locked" in e
        or "unable to lock" in e
        or "resource temporarily unavailable" in e
        or "another task is running" in e
        or "lock file" in e
        or "lockfile" in e
        or "locked by" in e
        or "timeout" in e
    )


def _is_dnf_like(dnf) -> bool:
    """Defensive: ensure DNF looks like list[list[dict]]."""
    if not isinstance(dnf, (list, tuple)):
        return False
    for term in dnf:
        if not isinstance(term, (list, tuple)):
            return False
        for atom in term:
            if not isinstance(atom, dict):
                return False
    return True

def _is_atom_like(atom) -> bool:
    if not isinstance(atom, dict):
        return False
    typ = (atom.get("typ") or atom.get("type") or "").strip()
    if not typ:
        return False
    mods = atom.get("mods", {})
    if mods is not None and not isinstance(mods, dict):
        return False
    return True


def _normalize_dnf_cached(dnf):
    """Normalize cached DNF to match parser types (tuple for single time)."""
    if not isinstance(dnf, (list, tuple)):
        return dnf
    for term in dnf:
        if not isinstance(term, (list, tuple)):
            continue
        for atom in term:
            if not isinstance(atom, dict):
                continue
            mods = atom.get("mods")
            if not isinstance(mods, dict):
                continue
            tval = mods.get("t")
            if isinstance(tval, list):
                if len(tval) == 2 and all(isinstance(x, int) for x in tval):
                    mods["t"] = (tval[0], tval[1])
                elif tval and all(
                    isinstance(x, list) and len(x) == 2 and all(isinstance(y, int) for y in x)
                    for x in tval
                ):
                    mods["t"] = [(x[0], x[1]) for x in tval]
    return dnf

def cache_load(key: str) -> dict | None:
    if not ENABLE_ANCHOR_CACHE:
        return None
    path = _cache_path(key)
    if not path:
        return None
    try:
        st = os.stat(path)
        if ANCHOR_CACHE_TTL and (time.time() - st.st_mtime) > ANCHOR_CACHE_TTL:
            return None
        with open(path, "rb") as f:
            blob = f.read()
        data = zlib.decompress(base64.b85decode(blob))
        obj = json.loads(data.decode("utf-8"))
        if isinstance(obj, dict) and "dnf" in obj:
            obj["dnf"] = _normalize_dnf_cached(obj.get("dnf"))
            if not _is_dnf_like(obj.get("dnf")):
                return None
        return obj
    except (OSError, ValueError, json.JSONDecodeError, zlib.error) as e:
        if os.environ.get("NAUTICAL_DIAG") == "1":
            diag(f"cache_load failed: {e}")
        return None

def cache_save(key: str, obj: dict) -> None:
    if not ENABLE_ANCHOR_CACHE:
        return
    data = json.dumps(obj, separators=(",", ":"), sort_keys=True).encode("utf-8")
    blob = base64.b85encode(zlib.compress(data, 9))
    path = _cache_path(key)
    if not path:
        return
    tmpf = None
    try:
        base = _cache_dir()
        if not base:
            return
        with _cache_lock(key) as locked:
            if not locked:
                return
            try:
                for name in os.listdir(base):
                    if name.startswith(f".{key}.") and name.endswith(".tmp"):
                        try:
                            os.unlink(os.path.join(base, name))
                        except Exception:
                            pass
            except Exception:
                pass
            fd, tmpf = tempfile.mkstemp(dir=base, prefix=f".{key}.", suffix=".tmp")
            try:
                os.fchmod(fd, 0o600)
            except Exception:
                pass
            try:
                written = 0
                while written < len(blob):
                    n = os.write(fd, blob[written:])
                    if n == 0:
                        raise OSError("write returned 0")
                    written += n
            finally:
                try:
                    os.close(fd)
                except Exception:
                    pass
            os.replace(tmpf, path)
    except (OSError, ValueError, json.JSONDecodeError, zlib.error) as e:
        if os.environ.get("NAUTICAL_DIAG") == "1":
            diag(f"cache_save failed: {e}")
    finally:
        if tmpf and os.path.exists(tmpf):
            try:
                os.unlink(tmpf)
            except Exception:
                pass

def cache_key_for_task(anchor_expr: str, anchor_mode: str) -> str:
    # Ephemeral canonical: never stored on the task, used only to build a stable key
    try:
        acf = build_acf(anchor_expr)
    except Exception:
        acf = (anchor_expr or "").strip()
    return _cache_key(acf, anchor_mode or "")


# ---- Core iterator over DNF ---------------------------------------------------
_NTH_RE  = re.compile(r"^(?:(\d)(?:st|nd|rd|th)|last)-(" + "|".join(_WD_ABBR) + r")$")

def _days_in_month(y:int, m:int) -> int:
    return calendar.monthrange(y, m)[1]

def _wd_idx(s: str) -> int | None:
    s = (s or "").strip().lower()
    if s in _WD_ABBR: return _WD_ABBR.index(s)
    try:
        n = int(s)
        if 1 <= n <= 7: return n-1
    except Exception:
        pass
    return None


@_ttl_lru_cache(maxsize=128)
def _wday_idx_any(s: str) -> int | None:
    """Weekday index for weekly specs.

    Accepts:
      - abbreviations: mon..sun
      - full names:    monday..sunday
      - numeric:       1..7 (Mon=1)
    """
    s = (s or "").strip().lower()
    if not s:
        return None
    if s in _WEEKDAYS:
        return _WEEKDAYS[s]
    return _wd_idx(s)


def _weekly_spec_to_wset(spec: str, mods: dict | None = None) -> set[int]:
    """Expand a weekly spec into a weekday set {0..6}.

    Supports canonical V2 ranges ("..").

    If the spec contains 'rand', treat it as the full weekday pool (or Mon–Fri
    if @bd/@wd is active). If 'rand' is combined with explicit tokens (should
    be rejected by strict validation), we return the union for resilience.
    """
    spec = _expand_weekly_aliases(spec)
    if not spec:
        return set()

    toks = _split_csv_lower(spec)
    out: set[int] = set()

    if any(t == "rand" for t in toks):
        pool = (
            {0, 1, 2, 3, 4}
            if ((mods or {}).get("bd") or (mods or {}).get("wd") is True)
            else {0, 1, 2, 3, 4, 5, 6}
        )
        out |= pool

    for tok in toks:
        if tok == "rand":
            continue
        if ".." in tok:
            a, b = tok.split("..", 1)
            ia, ib = _wday_idx_any(a), _wday_idx_any(b)
            if ia is None or ib is None:
                continue
            rng = (
                list(range(ia, ib + 1))
                if ia <= ib
                else (list(range(ia, 7)) + list(range(0, ib + 1)))
            )
            out.update(rng)
        else:
            i = _wday_idx_any(tok)
            if i is not None:
                out.add(i)

    return out

def _doms_for_weekly_spec(spec:str, y:int, m:int) -> set[int]:
    """Return DOMs in month (y,m) whose weekday matches any in spec (e.g., 'mon,thu' or 'mon..fri')."""
    spec = _expand_weekly_aliases(spec)
    if not spec: return set()
    allowed: set[int] = set()
    # expand tokens and ranges
    wset: set[int] = set()
    for tok in _split_csv_tokens(spec):
        if ".." in tok:
            a, b = tok.split("..", 1)
            ia, ib = _wd_idx(a), _wd_idx(b)
            if ia is None or ib is None: continue
            rng = list(range(ia, ib+1)) if ia <= ib else (list(range(ia,7))+list(range(0,ib+1)))
            wset.update(rng)
        else:
            i = _wd_idx(tok)
            if i is not None: wset.add(i)
    if not wset: return set()
    dim = _days_in_month(y,m)
    for d in range(1, dim+1):
        if date(y,m,d).weekday() in wset:
            allowed.add(d)
    return allowed

def _doms_for_monthly_token(tok: str, y:int, m:int) -> set[int]:
    """Support: 'rand' -> full month; '10..20'; '31'; '-1'; '2nd-mon'; 'last-fri'."""
    tok = (tok or "").strip().lower()
    if tok in _MONTHLY_ALIAS:
        tok = _MONTHLY_ALIAS[tok]
    dim = _days_in_month(y,m)
    if tok == "rand":
        return set(range(1, dim+1))
    # range a..b
    m2 = re.fullmatch(r"(\-?\d{1,2})\.\.(\-?\d{1,2})", tok)
    if m2:
        a, b = int(m2.group(1)), int(m2.group(2))
        if a < 0: a = dim + 1 + a  # -1 -> dim
        if b < 0: b = dim + 1 + b
        a = max(1, min(dim, a)); b = max(1, min(dim, b))
        lo, hi = (a,b) if a <= b else (b,a)
        return set(range(lo, hi+1))
    # single int (may be negative)
    if re.fullmatch(r"\-?\d{1,2}", tok):
        d = int(tok)
        if d < 0: d = dim + 1 + d
        if 1 <= d <= dim: return {d}
        return set()
    # nth/last weekday
    m3 = _NTH_RE.fullmatch(tok)
    if m3:
        nth_s, wd_s = m3.group(1), m3.group(2)
        wd = _wd_idx(wd_s)
        if wd is None: return set()
        days = [d for d in range(1, dim+1) if date(y,m,d).weekday() == wd]
        if nth_s:
            idx = int(nth_s)-1
            return {days[idx]} if 0 <= idx < len(days) else set()
        else:  # last-*
            return {days[-1]} if days else set()
    # unknown monthly token -> empty set (parser should have validated earlier)
    return set()

def _y_ranges_from_spec(spec: str) -> list[tuple[int,int,int,int]]:
    out = []
    for tok in _split_csv_lower(spec):

        # NEW: support 'rand-MM' → entire month (clamped later)
        m_randm = re.fullmatch(r"rand-(\d{2})", tok)
        if m_randm:
            mm = int(m_randm.group(1))
            if 1 <= mm <= 12:
                out.append((mm, 1, mm, 31))  # end will be clamped downstream
            continue

        m = re.fullmatch(r"(\d{2})-(\d{2})(?:\.\.(\d{2})-(\d{2}))?$", tok)
        if not m:
            continue
        a,b = int(m.group(1)), int(m.group(2))
        d1, m1 = _year_pair(a,b)
        if m.group(3):
            c,d = int(m.group(3)), int(m.group(4))
            d2, m2 = _year_pair(c,d)
        else:
            d2, m2 = d1, m1
        out.append((m1,d1,m2,d2))
    return out


def _doms_allowed_by_year(y:int, m:int, y_specs: list[str]) -> set[int]:
    """Return DOMs in month (y,m) that are inside ANY yearly range/day specified across all y atoms of the term."""
    if not y_specs:
        return set(range(1, _days_in_month(y,m)+1))
    # plain 'rand' means "no yearly restriction"
    if any((sp or "").strip().lower() == "rand" for sp in y_specs):
        return set(range(1, _days_in_month(y,m)+1))

    # Collect all ranges from all y atoms, then union the DOMs in (y,m)
    ranges = []
    for sp in y_specs:
        ranges.extend(_y_ranges_from_spec(sp))
    if not ranges:
        return set()
    dim = _days_in_month(y,m)
    allowed = set()
    for (m1,d1,m2,d2) in ranges:
        if m1 == m2:
            if m == m1:
                lo, hi = max(1, d1), min(dim, d2)
                allowed.update(range(lo, hi+1))
        else:
            # spans months
            if m < m1 or m > m2:
                continue
            if m == m1:
                allowed.update(range(max(1,d1), dim+1))
            elif m == m2:
                allowed.update(range(1, min(dim,d2)+1))
            elif m1 < m < m2:
                allowed.update(range(1, dim+1))
    return allowed

def _choose_rand_dom(y:int, m:int, doms: set[int]) -> int | None:
    """Deterministic pick of one day from 'doms' using WRAND_SALT."""
    if not doms:
        return None
    pool = sorted(doms)
    h = hashlib.sha256(f"{WRAND_SALT}|{y:04d}-{m:02d}".encode("utf-8")).digest()
    idx = int.from_bytes(h[:8], "big") % len(pool)
    return pool[idx]

def _next_for_and(term: list[dict], ref_d: date, seed: date) -> date:
    """
    Find the next date > ref_d satisfying ALL atoms in 'term'.
    Rand-aware: if the term contains m:rand and any y:, choose the random
    day from the intersection of ALL constraints for each candidate month.
    Otherwise, fall back to the fast alignment loop.
    """
    # Detect rand + yearly presence
    has_m_rand = any((a.get("typ") or a.get("type")) == "m" and "rand" in str(a.get("spec") or "").lower()
                     for a in term)
    y_specs = [str(a.get("spec") or "") for a in term if (a.get("typ") or a.get("type")) == "y"]

    if has_m_rand and y_specs:
        # Month-by-month scan, pick deterministic random from allowed set
        y, m = ref_d.year, ref_d.month
        # step starts at ref day + 1
        probe = ref_d + timedelta(days=1)
        for _ in range(60):  # scan up to 5 years (60 months)
            y, m = probe.year, probe.month
            dim = _days_in_month(y,m)

            # Start with all days in month
            allowed = set(range(1, dim+1))

            # Intersect with yearly windows
            allowed &= _doms_allowed_by_year(y,m, y_specs)
            if not allowed:
                # jump to first day of next month
                probe = (date(y,m,1) + timedelta(days=dim))  # first day next month
                continue

            # Intersect with all monthly atoms (except 'rand', which we resolve after)
            for a in term:
                typ = (a.get("typ") or a.get("type") or "").lower()
                spec = str(a.get("spec") or "")
                if typ != "m":
                    continue
                toks = _split_csv_lower(spec)
                if not toks:
                    continue
                # union across tokens in this atom, then intersect across atoms
                u: set[int] = set()
                for tok in toks:
                    if tok == "rand":
                        # defer to final choose-from-allowed
                        u.update(range(1, dim+1))
                    else:
                        u.update(_doms_for_monthly_token(tok, y, m))
                allowed &= u
                if not allowed:
                    break
            if not allowed:
                probe = (date(y,m,1) + timedelta(days=dim))
                continue

            # Intersect with weekly atoms
            for a in term:
                typ = (a.get("typ") or a.get("type") or "").lower()
                if typ != "w":
                    continue
                spec = str(a.get("spec") or "")
                wdom = _doms_for_weekly_spec(spec, y, m)
                if not wdom:
                    allowed = set()
                else:
                    allowed &= wdom
                if not allowed:
                    break
            if not allowed:
                probe = (date(y,m,1) + timedelta(days=dim))
                continue

            # Choose deterministic random day from allowed set
            pick = _choose_rand_dom(y, m, allowed)
            if pick is None:
                probe = (date(y,m,1) + timedelta(days=dim))
                continue

            cand = date(y, m, pick)
            if cand > ref_d:
                return cand
            # else move to next month if pick ≤ ref
            probe = (date(y,m,1) + timedelta(days=dim))
        # fallback
        return ref_d + timedelta(days=365)

    # -------- Fast path (no rand+yearly combo) ----------
    probe = ref_d
    for _ in range(MAX_ANCHOR_ITER):
        cands = [next_after_atom_with_mods(atom, probe, seed) for atom in term]
        target = max(cands)
        if target <= probe:
            if os.environ.get("NAUTICAL_DIAG") == "1":
                _warn_once_per_day(
                    "next_for_and_no_progress",
                    "[nautical] _next_for_and made no progress; failing fast. Check anchor spec.",
                )
            raise ParseError("Anchor evaluation made no forward progress; check anchor spec.")
        # Verify target matches all atoms (handles modifiers like @bd)
        if all(atom_matches_on(atom, target, seed) for atom in term):
            return target
        probe = target
    if os.environ.get("NAUTICAL_DIAG") == "1":
        _warn_once_per_day(
            "next_for_and_fallback",
            f"[nautical] _next_for_and fallback after {MAX_ANCHOR_ITER} iterations.",
        )
    return ref_d + timedelta(days=365)


def _next_for_or(dnf: list[list[dict]], ref_d: date, seed: date) -> date:
    best = None
    for term in dnf:
        d = _next_for_and(term, ref_d, seed)
        if d and d > ref_d and (best is None or d < best):
            best = d
    return best or (ref_d + timedelta(days=365))

# ---- Public precompute --------------------------------------------------------

def precompute_hints(dnf: list[list[dict]],
                     start_dt: datetime | None = None,
                     anchor_mode: str = "ALL",
                     rand_seed: str | None = None,
                     k_next: int = 24,
                     sample_days_for_year: int = 366) -> dict:

    # Operate in local dates; let hooks add times if they prefer.
    today = datetime.now().date()
    start_d = (start_dt.date() if isinstance(start_dt, datetime) else start_dt) or today

    # If any rand atom exists, the optimized _next_for_or/_next_for_and path is not sufficient
    # (notably for yearly rand windows like y:rand-07). Use next_after_expr which is authoritative.
    has_rand = any(
        "rand" in str((a.get("spec") or "")).lower()
        for term in (dnf or [])
        for a in (term or [])
    )

    out_next: list[str] = []
    ref = start_d

    # Keep /N gating stable relative to preview start.
    default_seed = ref
    seed_base = rand_seed or "preview"

    safety_limit = 366 * 5
    steps = 0
    while len(out_next) < k_next and steps < safety_limit:
        if has_rand:
            nxt, _ = next_after_expr(dnf, ref, default_seed=default_seed, seed_base=seed_base)
        else:
            nxt = _next_for_or(dnf, ref, default_seed)

        if not nxt or nxt <= ref:
            break

        out_next.append(nxt.isoformat() + "T00:00")
        ref = nxt + timedelta(days=1)
        steps += 1

    # Estimate per-year by scanning ~1 year ahead
    year_hits = 0
    first_hit = last_hit = ""
    ref = today
    steps = 0
    seen: set[date] = set()

    while steps < sample_days_for_year:
        if has_rand:
            nxt, _ = next_after_expr(dnf, ref, default_seed=default_seed, seed_base=seed_base)
        else:
            nxt = _next_for_or(dnf, ref, default_seed)

        if not nxt or nxt <= ref:
            break

        iso_s = nxt.isoformat() + "T00:00"
        if not first_hit:
            first_hit = iso_s
        last_hit = iso_s

        if nxt not in seen:
            seen.add(nxt)
            year_hits += 1

        ref = nxt + timedelta(days=1)
        steps += 1

    per_year = {"est": year_hits, "first": first_hit, "last": last_hit}
    limits = {"stop": "none", "max_left": 0, "until": ""}
    rand_preview = out_next[:10]

    return {
        "next_dates": out_next,
        "per_year": per_year,
        "limits": limits,
        "rand_preview": rand_preview,
    }


# ───────────────── Cache writer ───────────────── 

def build_and_cache_hints(anchor_expr: str,
                          anchor_mode: str = "ALL",
                          default_due_dt=None) -> dict:
    key = cache_key_for_task(anchor_expr, anchor_mode)
    cached = cache_load(key)
    if cached:
        if "dnf" in cached:
            cached["dnf"] = _normalize_dnf_cached(cached.get("dnf"))
            if not _is_dnf_like(cached.get("dnf")):
                cached = None
        return cached

    dnf = validate_anchor_expr_strict(anchor_expr)
    natural = describe_anchor_expr(anchor_expr, default_due_dt=default_due_dt)
    hints = precompute_hints(dnf, start_dt=default_due_dt, anchor_mode=anchor_mode)

    payload = {
        "meta": {"created": int(time.time()),
                 "cfg": {"fmt": ANCHOR_YEAR_FMT, "salt": WRAND_SALT, "tz": LOCAL_TZ_NAME, "hol": HOLIDAY_REGION}},
        "dnf": dnf,
        "natural": natural,
        **hints,
    }
    cache_save(key, payload)
    return payload


# ───────────────── Quarter helpers ─────────────────
# Recognize full-month tokens like '01-03..31-03'
_FULL_MONTH_RE = re.compile(r"^01-(\d{2})\.\.(\d{2})-(\d{2})$")
# Recognize day-only tokens like '31-03'
_DAY_ONLY_RE = re.compile(r"^(\d{2})-(\d{2})$")

# Month → quarter (first month of each quarter)
_Q_BY_FIRST_MONTH = {1: 1, 4: 2, 7: 3, 10: 4}
# Quarter first-month ranges as produced by the rewriter
_Q_FIRST_MONTH_TOKEN = {
    1: "01-01..31-01",  # Jan
    2: "01-04..30-04",  # Apr
    3: "01-07..31-07",  # Jul
    4: "01-10..31-10",  # Oct
}
# Quarter start day tokens
_Q_START_DAY = {1: "01-01", 2: "01-04", 3: "01-07", 4: "01-10"}
# Quarter end day tokens
_Q_END_DAY = {1: "31-03", 2: "30-06", 3: "30-09", 4: "31-12"}


def _yearly_tokens(term):
    out = []
    for a in term:
        if (a.get("typ") or a.get("type") or "").lower() == "y":
            spec = (a.get("spec") or a.get("value") or "").lower()
            out.extend(_split_csv_tokens(spec))
    return out


def _monthly_tokens(term):
    out = []
    for a in term:
        if (a.get("typ") or a.get("type") or "").lower() == "m":
            spec = (a.get("spec") or a.get("value") or "").lower()
            out.extend(_split_csv_tokens(spec))
    return out


# def _extract_single_nth_weekday(term):
#     """Return (k, wd_str) if exactly one monthly nth-weekday token is present; else None."""
#     from itertools import chain

#     toks = _monthly_tokens(term)
#     if len(toks) != 1:
#         return None
#     m = _nth_weekday_re.match(toks[0]) 
#     if not m:
#         return None
#     n_raw, wd = m.group(1), m.group(2)
#     if n_raw == "last":
#         k = -1
#     else:
#         k = int(re.sub(r"(st|nd|rd|th)$", "", n_raw))
#         if k == 0 or abs(k) > 5:
#             return None
#     return k, wd  # wd is 'mon'..'sun'


def _quarters_from_first_month_tokens(y_toks):
    """Return sorted unique quarters implied by first-month full ranges."""
    qs = []
    for tok in y_toks:
        if tok in _Q_FIRST_MONTH_TOKEN.values():
            # reverse map: token -> quarter
            q = {v: k for k, v in _Q_FIRST_MONTH_TOKEN.items()}[tok]
            qs.append(q)
    return sorted(set(qs))


def _quarters_from_start_day_tokens(y_toks):
    qs = []
    for tok in y_toks:
        if tok in _Q_START_DAY.values():
            q = {v: k for k, v in _Q_START_DAY.items()}[tok]
            qs.append(q)
    return sorted(set(qs))


def _quarters_from_end_day_tokens(y_toks):
    qs = []
    for tok in y_toks:
        if tok in _Q_END_DAY.values():
            q = {v: k for k, v in _Q_END_DAY.items()}[tok]
            qs.append(q)
    return sorted(set(qs))


def _format_quarter_set(qs):
    """Return ('each quarter' | 'Q2' | 'Q1–Q2' | 'Q1 and Q3')."""
    if not qs:
        return None
    if qs == [1, 2, 3, 4]:
        return "each quarter"
    # contiguous?
    if max(qs) - min(qs) + 1 == len(qs):
        if len(qs) == 2:
            return f"Q{qs[0]}–Q{qs[1]}"
        return ", ".join(f"Q{x}" for x in qs[:-1]) + f" and Q{qs[-1]}"
    # non-contiguous
    if len(qs) == 1:
        return f"Q{qs[0]}"
    return ", ".join(f"Q{x}" for x in qs[:-1]) + f" and Q{qs[-1]}"


def _rewrite_quarter_spec_mode(spec: str, mode: str, meta_out: dict | None = None) -> str:
    """
    Rewrite q1..q4 tokens in a yearly spec into concrete y:* tokens.

    Important:
      - quarter_start / first_month rewrite to the *first month window* (Jan/Apr/Jul/Oct)
      - quarter_end rewrites to the *last month window* (Mar/Jun/Sep/Dec)

    Rationale: quarter_end MUST be a month-window (not a fixed day like 31-12),
    otherwise selectors like m:-1bd cannot express "last business day of the quarter".

    If meta_out is provided, it is filled with { rewritten_token -> quarter_note } so
    the Natural layer can disclose interpretation (e.g. "Oct each year (Q4 start month)").
    """
    if not spec:
        return spec

    qmap: dict[str, str] = {}

    def _first_month_window(q: int) -> str:
        mm = {1: 1, 2: 4, 3: 7, 4: 10}[q]
        return _tok_range(1, mm, _static_month_last_day(mm), mm)

    def _last_month_window(q: int) -> str:
        mm = {1: 3, 2: 6, 3: 9, 4: 12}[q]
        return _tok_range(1, mm, _static_month_last_day(mm), mm)

    def _mid_month_window(q: int) -> str:
        mm = {1: 2, 2: 5, 3: 8, 4: 11}[q]
        return _tok_range(1, mm, _static_month_last_day(mm), mm)

    def _pos_month_window(q: int, pos: str) -> str | None:
        mm = _QUARTER_POS_MONTH.get(q, {}).get(pos)
        if not mm:
            return None
        return _tok_range(1, mm, _static_month_last_day(mm), mm)

    def _emit(q: int) -> str:
        if mode == "quarter_end":
            return _last_month_window(q)
        if mode == "quarter_mid":
            return _mid_month_window(q)
        if mode == "quarter_window":
            start = {1: 1, 2: 4, 3: 7, 4: 10}[q]
            end = {1: 3, 2: 6, 3: 9, 4: 12}[q]
            return _tok_range(1, start, 31, end)
        # quarter_start and first_month both select the quarter's first month window
        return _first_month_window(q)

    def _note(q: int) -> str:
        if mode == "quarter_end":
            return f"Q{q} end month"
        if mode == "quarter_mid":
            return f"Q{q} mid month"
        if mode == "quarter_start":
            return f"Q{q} start month"
        if mode == "quarter_window":
            return f"Q{q} window"
        # default (no monthly disambiguator present)
        return f"Q{q} first month"

    out = []
    toks = _split_csv_lower(spec)

    for tok in toks:
        m = re.fullmatch(r"q([1-4])([sme])", tok)
        if m:
            q = int(m.group(1))
            pos = m.group(2)
            w = _pos_month_window(q, pos)
            if w:
                out.append(w)
                pos_note = {"s": "start", "m": "mid", "e": "end"}[pos]
                qmap[w] = f"Q{q} {pos_note} month"
            continue

        m = re.fullmatch(r"q([1-4])([sme])\.\.q([1-4])([sme])", tok)
        if m:
            qa, qb = int(m.group(1)), int(m.group(3))
            posa, posb = m.group(2), m.group(4)
            if posa != posb:
                out.append(tok)
                continue
            if qa > qb:
                out.append(tok)  # leave as-is so the validator can raise nicely elsewhere
            else:
                for q in range(qa, qb + 1):
                    w = _pos_month_window(q, posa)
                    if w:
                        out.append(w)
                        pos_note = {"s": "start", "m": "mid", "e": "end"}[posa]
                        qmap[w] = f"Q{q} {pos_note} month"
            continue

        m = re.fullmatch(r"q([1-4])", tok)
        if m:
            q = int(m.group(1))
            w = _emit(q)
            out.append(w)
            qmap[w] = _note(q)
            continue

        m = re.fullmatch(r"q([1-4])\.\.q([1-4])", tok)
        if m:
            qa, qb = int(m.group(1)), int(m.group(2))
            if qa > qb:
                out.append(tok)  # leave as-is so the validator can raise nicely elsewhere
            else:
                for q in range(qa, qb + 1):
                    w = _emit(q)
                    out.append(w)
                    qmap[w] = _note(q)
            continue

        out.append(tok)

    # De-dup while preserving order
    seen, dedup = set(), []
    for t in out:
        if t not in seen:
            dedup.append(t)
            seen.add(t)

    if meta_out is not None and qmap:
        meta_out.update(qmap)

    return ",".join(dedup)



def _rewrite_quarters_in_context(dnf):
    """
    Walk DNF (list of AND-terms). For each term:
      - if it contains quarter aliases in a y:* atom (q1..q4, q1..q4 ranges),
      - choose rewrite mode based on monthly context,
      - enforce strict rules to prevent ambiguous quarter meaning,
      - rewrite y:* and attach a per-atom _qmap (rewritten_token -> note) for Natural output.

    Strict rule:
      If a term uses y:q* together with m:* then m:* must be a SINGLE token that
      unambiguously indicates *quarter start month* or *quarter end month*.

      Start-month selectors (allowed):
        - 1
        - 1bd
        - 1mon / 1st-mon (first weekday-of-month)

      End-month selectors (allowed):
        - -N or -Nbd (e.g. -1, -1bd, -2bd)
        - last-fri (or last<weekday>)
        - -Nfri (or -N<weekday>)

      Anything else (e.g. m:15, m:2bd, m:rand, m:1,15, multiple m atoms) is rejected.
      For mid-month semantics inside a quarter, use explicit months (e.g. y:dec).
    """

    def _has_quarter_tokens(spec: str) -> bool:
        for t in _split_csv_lower(spec):
            if re.fullmatch(r"q[1-4][sme]?(?:\.\.q[1-4][sme]?)?", t):
                return True
        return False

    def _has_plain_quarter_tokens(spec: str) -> bool:
        for t in _split_csv_lower(spec):
            if re.fullmatch(r"q[1-4](?:\.\.q[1-4])?", t):
                return True
        return False

    def _is_start_month_selector(tok: str) -> bool:
        t = (tok or "").strip().lower()
        if t in ("1", "1bd"):
            return True
        m = _nth_weekday_re.match(t)
        if m:
            n_raw = (m.group(1) or "").lower()
            return n_raw in ("1", "1st")
        return False

    def _is_end_month_selector(tok: str) -> bool:
        t = (tok or "").strip().lower()
        # -N
        if re.fullmatch(r"-\d+", t):
            return True
        # -Nbd
        m_bd = _bd_re.match(t)
        if m_bd and int(m_bd.group(1)) < 0:
            return True
        # last-fri / -Nfri
        m = _nth_weekday_re.match(t)
        if m:
            n_raw = (m.group(1) or "").lower()
            return n_raw == "last" or n_raw.startswith("-")
        return False

    for term in dnf:
        y_atoms = [a for a in term if (a.get("typ") or a.get("type") or "").lower() == "y"]
        if not y_atoms:
            continue

        # Only engage quarter logic if at least one yearly atom actually contains q-tokens
        if not any(_has_quarter_tokens((a.get("spec") or a.get("value") or "").lower()) for a in y_atoms):
            continue

        mode = "first_month"  # default when no monthly disambiguator is present

        m_atoms = [a for a in term if (a.get("typ") or a.get("type") or "").lower() == "m"]
        has_plain_quarters = any(
            _has_plain_quarter_tokens((a.get("spec") or a.get("value") or "").lower())
            for a in y_atoms
        )
        if m_atoms and has_plain_quarters:
            if len(m_atoms) != 1:
                raise ParseError(
                    "Quarter aliases (y:q1..q4) cannot be combined with multiple monthly atoms in the same term. "
                    "Use a single m:* selector or replace y:q* with explicit months (e.g. y:oct..dec)."
                )

            mspec = (m_atoms[0].get("spec") or m_atoms[0].get("value") or "").strip().lower()
            mspec = _expand_monthly_aliases(mspec)
            if mspec == "rand":
                raise ParseError(
                    "Quarter aliases (y:q1..q4) cannot be combined with m:rand. "
                    "Use explicit months if you need randomness within a quarter-like window."
                )

            mtoks = _split_csv_tokens(mspec)
            if len(mtoks) != 1:
                raise ParseError(
                    "Quarter aliases (y:q1..q4) require a single monthly selector token when used with m:*. "
                    "Examples: m:1bd + y:q4 (start month) OR m:-1bd + y:q4 (end month). "
                    "If you meant a specific month of the quarter, use y:q4s/y:q4m/y:q4e."
                )

            mt = mtoks[0]
            if _is_end_month_selector(mt):
                mode = "quarter_end"
            elif _is_start_month_selector(mt):
                mode = "quarter_start"
            else:
                mode = "quarter_window"
                raise ParseError(
                    "Quarter aliases (y:q1..q4) paired with m:* are ambiguous here. "
                    "Use y:qNs/y:qNm/y:qNe to target start/mid/end month, or use explicit months (e.g. y:oct..dec)."
                )

        # Rewrite each yearly atom; attach metadata so Natural can disclose interpretation
        for ya in y_atoms:
            spec = (ya.get("spec") or ya.get("value") or "").lower()
            if not _has_quarter_tokens(spec):
                continue

            qmeta: dict[str, str] = {}
            ya["spec"] = _rewrite_quarter_spec_mode(spec, mode, meta_out=qmeta)
            if qmeta:
                ya["_qmap"] = qmeta

    return dnf


def _rewrite_year_month_aliases_in_dnf(dnf: list[list[dict]]) -> list[list[dict]]:
    """
    In-place rewrite for y: specs:
      - 'y:apr'           → 'y:04-01..04-31' (per FMT)
      - 'y:jan..jun'      → 'y:01-01..06-31'
      - 'y:04'            → 'y:04-01..04-31'
      - 'y:04..06'        → 'y:04-01..06-31'
      - mixed 'y:apr..06' → 'y:04-01..06-31' etc.
    Leaves standard 'DD-MM..DD-MM' and 'rand', 'rand-MM' as-is.
    """
    for term in dnf:
        for atom in term:
            if (atom.get("typ") or atom.get("type") or "").lower() != "y":
                continue
            spec = (atom.get("spec") or atom.get("value") or "").strip().lower()
            if not spec:
                continue

            toks_in = _split_csv_tokens(spec)
            toks_out = []

            for tok in toks_in:
                # Pass through already-standard forms and rand variants
                if re.fullmatch(r"\d{2}-\d{2}(?:\.\.\d{2}-\d{2})?$", tok) or tok == "rand" or re.fullmatch(r"rand-\d{2}", tok):
                    toks_out.append(tok)
                    continue

                # Single month alias (name or MM)
                m_single = _month_from_alias(tok)
                if m_single:
                    toks_out.append(_year_full_month_range_token(m_single))
                    continue

                # Month:Month range (name/name, name/MM, MM/name, MM/MM)
                m = re.fullmatch(r"([a-z]{3}|\d{2})\.\.([a-z]{3}|\d{2})", tok)
                if m:
                    m1 = _month_from_alias(m.group(1))
                    m2 = _month_from_alias(m.group(2))
                    if m1 and m2:
                        toks_out.append(_year_full_months_span_token(m1, m2))
                        continue
                    # fall-through if bad alias; validator will flag later

                # Unknown token -> keep as-is; strict validator/linter can surface a friendly error
                toks_out.append(tok)

            atom["spec"] = ",".join(toks_out)

    return dnf



def _bd_shift_from_term(term) -> str | None:
    """Return 'pbd' (previous) or 'nbd' (next) if any atom has that modifier."""
    saw_p = False
    saw_n = False
    for a in term:
        mods = a.get("mods") or {}
        if mods.get("pbd"):
            saw_p = True
        if mods.get("nbd"):
            saw_n = True
    # If both appear, prefer explicit 'previous' (deterministic; you can swap if you prefer)
    if saw_p and not saw_n:
        return "pbd"
    if saw_n and not saw_p:
        return "nbd"
    if saw_p and saw_n:
        return "pbd"
    return None


def _bd_shift_suffix(kind: str) -> str:
    """Kind is 'pbd' or 'nbd'. Returns the human phrase that will be splice in."""
    return (
        " if business day; otherwise the previous business day"
        if kind == "pbd"
        else " if business day; otherwise the next business day"
    )


# ───────────────── Natural language for anchor ─────────────────

_WDNAME = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}
_MONTH_ABBR = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]
_MONTH_FULL = list(month_name)
_WD_INDEX = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
_WD_FULL = {
    "mon": "Monday",
    "tue": "Tuesday",
    "wed": "Wednesday",
    "thu": "Thursday",
    "fri": "Friday",
    "sat": "Saturday",
    "sun": "Sunday",
}


def _ordinal(n: int) -> str:
    n = int(n)
    if 10 <= n % 100 <= 20:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def _term_collect_mods(term: list) -> dict:
    """Collapse per-atom mods into one dict for prose purposes (last writer wins)."""
    merged = {}
    for a in term:
        mods = a.get("mods") or {}
        for k, v in mods.items():
            merged[k] = v
    return merged


def _fmt_hhmm_for_term(term: list, default_due_dt):
    """Return HH:MM (or 'HH:MM, HH:MM, ...') for prose only when explicitly set via @t."""
    mods = _term_collect_mods(term)
    tmod = mods.get("t")
    if isinstance(tmod, tuple):
        return f"{tmod[0]:02d}:{tmod[1]:02d}"
    if isinstance(tmod, list):
        parts = []
        for x in tmod:
            if isinstance(x, tuple) and len(x) == 2:
                parts.append(f"{x[0]:02d}:{x[1]:02d}")
        return ", ".join(parts) if parts else None
    if isinstance(tmod, str) and tmod:
        return tmod
    return None


def _fmt_weekdays_list(spec: str) -> str:
    spec = _expand_weekly_aliases(spec)
    tokens = _split_csv_lower(spec)
    if not tokens:
        return ""

    plural = {
        0: "Mondays",
        1: "Tuesdays",
        2: "Wednesdays",
        3: "Thursdays",
        4: "Fridays",
        5: "Saturdays",
        6: "Sundays",
    }

    names: list[str] = []
    for t in tokens:
        if t == "rand":
            names.append("one random day each week")
            continue

    # Range tokens: mon..fri (canonical)
        if ".." in t:
            a, b = t.split("..", 1)
            ia, ib = _wday_idx_any(a), _wday_idx_any(b)
            if ia is None or ib is None:
                continue
            if ia == ib:
                names.append(plural[ia])
            else:
                names.append(f"{plural[ia]} through {plural[ib]}")
            continue

        i = _wday_idx_any(t)
        if i is not None:
            names.append(plural[i])

    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " or " + names[-1]


def _fmt_monthly_atom(spec: str) -> str:
    s = (spec or "").lower().strip()
    if s in _MONTHLY_ALIAS:
        s = _MONTHLY_ALIAS[s]
    if s == "rand":
        return "one random day each month"

    m = _safe_match(_nth_wd_re, s)
    if m:
        idx, wd = m.group(1), m.group(2)
        name = _WDNAME[_WD_INDEX[wd]]  # Title Case (Monday, Friday…)
        if idx == "last":
            return f"the last {name} of each month"
        k = int(re.sub(r"(st|nd|rd|th)$", "", idx))
        if k < 0:
            return f"the {_ordinal(abs(k))} last {name} of each month"
        return f"the {_ordinal(k)} {name} of each month"
    m = _bd_re.match(s)
    if m:
        k = int(m.group(1))
        if k > 0:
            return f"the {_ordinal(k)} business day of each month"
        if k == -1:
            return "the last business day of each month"
        return f"the {_ordinal(abs(k))} last business day of each month"
    if ".." in s:
        a, b = s.split("..", 1)
        try:
            ai = int(a)
            bi = int(b)
            if ai > 0 and bi > 0:
                return f"days {ai}–{bi} of each month"

            def dword(x):
                return (
                    "last day"
                    if x == -1
                    else (_ordinal(x) if x > 0 else f"{_ordinal(abs(x))} last day")
                )

            return f"days {dword(ai)}–{dword(bi)} of each month"
        except:
            pass
    try:
        k = int(s)
        if k == -1:
            return "the last day of each month"
        if k < 0:
            return f"the {_ordinal(abs(k))} last day of each month"
        return f"the {_ordinal(k)} day of each month"
    except:
        return f"[unknown monthly token '{spec}']"


def _fmt_md(d: int, m: int) -> str:
    fmt = (globals().get("ANCHOR_YEAR_FMT") or "DM").upper()
    name = _MONTH_ABBR[m - 1]
    return f"{d} {name}" if fmt == "DM" else f"{name} {d}"


def _is_full_month(d1, m1, d2, m2) -> int | None:
    """Return month number if token covers the whole month, else None.
    Accepts 28..31 as 'end of month' (Feb handled leniently)."""
    if m1 != m2 or d1 != 1:
        return None
    return m1 if 28 <= d2 <= 31 else None


def _fmt_yearly_atom(tok: str) -> str:
    s = (tok or "").strip().lower()

    # NEW: yearly random phrasing
    if s == "rand":
        return "one random day each year"

    m_randm = _rand_mm_re.fullmatch(s)
    if m_randm:
        mm = int(m_randm.group(1))
        if 1 <= mm <= 12:
            return f"one random day in {_MONTH_ABBR[mm-1]} each year"
        # fall through to generic handling if somehow invalid

    # Existing numeric handling (single or range), respecting ANCHOR_YEAR_FMT
    m = _md_range_re.fullmatch(s)
    if not m:
        return tok

    def _pair(a: int, b: int) -> tuple[int, int]:
        # returns (day, month) according to current FMT
        return (b, a) if _yearfmt() == "MD" else (a, b)

    a, b = int(m.group(1)), int(m.group(2))
    if m.group(3):
        c, d = int(m.group(3)), int(m.group(4))
        d1, m1 = _pair(a, b)
        d2, m2 = _pair(c, d)
        # full-month?
        if m1 == m2 and d1 == 1 and 28 <= d2 <= 31:
            # Example: '04-01..04-30' (or DM equivalent) → "Apr each year"
            return f"{_MONTH_ABBR[m1-1]} each year"
        if m1 == m2:
            # Same month, day range
            if _yearfmt() == "DM":
                return f"{d1}\u2013{d2} {_MONTH_ABBR[m1-1]} each year"
            else:
                return f"{_MONTH_ABBR[m1-1]} {d1}\u2013{d2} each year"
        # Cross-month range
        # If both ends cover full months (1..31), prefer "Jan–Jun each year".
        if d1 == 1 and 28 <= d2 <= 31:
            return f"{_MONTH_ABBR[m1-1]}\u2013{_MONTH_ABBR[m2-1]} each year"

        if _yearfmt() == "DM":
            left = f"{d1} {_MONTH_ABBR[m1-1]}"
            right = f"{d2} {_MONTH_ABBR[m2-1]}"
        else:
            left = f"{_MONTH_ABBR[m1-1]} {d1}"
            right = f"{_MONTH_ABBR[m2-1]} {d2}"
        return f"{left}\u2013{right} each year"
    else:
        # Single day
        d1, m1 = _pair(a, b)
        # Special-case Feb 29 phrasing remains
        if m1 == 2 and d1 == 29:
            return "Feb 29 each leap year"
        if _yearfmt() == "DM":
            return f"{d1} {_MONTH_ABBR[m1-1]} each year"
        else:
            return f"{_MONTH_ABBR[m1-1]} {d1} each year"




def describe_anchor_term(term: list, default_due_dt=None) -> str:

    def _ordinal(n: int) -> str:
        if 10 <= (n % 100) <= 20:
            suf = "th"
        else:
            suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suf}"

    def _parse_monthly_tokens(spec: str):
        return _split_csv_lower(spec)

    def _is_pure_nth_weekday_spec(spec: str):
        toks = _parse_monthly_tokens(spec)
        if not toks:
            return False, []
        out = []
        for t in toks:
            m = _safe_match(_nth_wd_re, t)
            if not m:
                return False, []
            n_raw, wd = m.group(1), m.group(2)
            if n_raw == "last":
                k = -1
            else:
                k = int(re.sub(r"(st|nd|rd|th)$", "", n_raw))
            out.append((k, wd))
        return True, out

    def _is_pure_dom_spec(spec: str):
        toks = _parse_monthly_tokens(spec)
        if not toks:
            return False, []
        out = []
        for t in toks:
            if not t.isdigit():
                return False, []
            d = int(t)
            if d < 1 or d > 31:
                return False, []
            out.append(d)
        return True, out

    def _single_full_month_from_yearly_spec(spec: str):
        m = _year_range_colon_re.match(str(spec or "").strip())
        if not m:
            return None
        d1, m1, d2, m2 = map(int, m.groups())
        if m1 != m2 or d1 != 1:
            return None
        if d2 < 28 or d2 > 31:
            return None
        return m1  # 1..12

    # ---- business-day roll helpers (pbd/nbd live in mods['roll']) ----
    def _term_roll_shift(term) -> str | None:
        saw = set()
        for a in term:
            roll = (a.get("mods") or {}).get("roll")
            if roll in ("nw", "pbd", "nbd"):
                saw.add(roll)
        if "nw" in saw:
            return "nw"
        if "pbd" in saw:
            return "pbd"
        if "nbd" in saw:
            return "nbd"
        return None

    # plain 'bd' filter present?
    def _term_bd_filter(term) -> bool:
        return any((a.get("mods") or {}).get("bd") for a in term)

    def _roll_suffix(roll: str) -> str:
        if roll == "pbd":
            return " if business day; otherwise the previous business day"
        if roll == "nbd":
            return " if business day; otherwise the next business day"
        if roll == "nw":
            return " if business day; otherwise the nearest business day (Fri if Saturday, Mon if Sunday)"
        return ""

    def _bd_suffix(roll: str) -> str:
        return (
            " if business day; otherwise the previous business day"
            if roll == "pbd"
            else " if business day; otherwise the next business day"
        )

    def _inject_schedule_suffixes(txt: str, term) -> str:
        """Add either roll suffix (pbd/nbd/nw) or bd-filter suffix (if no roll)."""
        roll = _term_roll_shift(term)
        if roll:
            suffix = _roll_suffix(roll)
        elif _term_bd_filter(term):
            suffix = " only if a business day (skipped if weekend)"
        else:
            suffix = ""

        if not suffix:
            return txt

        targets = [
            "the last day of each month",
            "the first day of each month",
            "the last day of the month",
            "the first day of the month",
            "the last day of each quarter",
            "the first day of each quarter",
        ]
        for t in targets:
            if t in txt:
                return txt.replace(t, t + suffix)

        if " at " in txt:
            head, sep, tail = txt.partition(" at ")
            return f"{head}{suffix} at {tail}"
        return txt + suffix

    # ---------- Build ----------
    parts = []
    m_parts = []
    y_parts = []
    w_phrase = None
    interval_prefix = None
    bd_filter = False
    suppress_tail = False  # when True, return only prefix (+ optional time)

    wk_ival = mo_ival = yr_ival = 1
    monthly_specs = []
    yearly_specs = []

    for a in term:
        typ = (a.get("typ") or a.get("type") or "").lower()
        spec = str(a.get("spec") or a.get("value") or "").strip().lower()
        ival = int(a.get("ival") or a.get("intv") or 1)

        if typ == "w":
            wk_ival = max(wk_ival, ival)
            w_phrase = _fmt_weekdays_list(spec)
            if wk_ival > 1 and spec == "rand":
                # Prefer: "one random day every 4 weeks"
                w_phrase = f"one random day every {wk_ival} weeks"

        elif typ == "m":
            mo_ival = max(mo_ival, ival)
            monthly_specs.append(spec)
            tokens = _split_csv_tokens(spec)
            for tok in tokens:
                m_parts.append(_fmt_monthly_atom(tok))

        elif typ == "y":
            yr_ival = max(yr_ival, ival)
            yearly_specs.append(spec)

            qmap = a.get("_qmap") or {}

            for tok in _split_csv_tokens(spec):
                phr = _fmt_yearly_atom(tok)
                if phr and qmap and tok in qmap and not phr.startswith("one random day"):
                    phr = f"{phr} ({qmap[tok]})"
                y_parts.append(phr)


        mods = a.get("mods") or {}
        bd_filter = bd_filter or bool(mods.get("bd") or (mods.get("wd") is True))

    # ---------- Fused phrasing: "2nd Monday of March every 2 years" ----------
    if len(monthly_specs) == 1 and len(yearly_specs) == 1:
        mspec = monthly_specs[0]
        yspec = yearly_specs[0]
        is_nth, pairs = _is_pure_nth_weekday_spec(mspec)
        fuse_month = _single_full_month_from_yearly_spec(yspec)
        if is_nth and fuse_month and len(pairs) == 1:
            k, wd = pairs[0]
            if k < 0:
                k_txt = "last" if k == -1 else f"{_ordinal(abs(k))} last"
            else:
                k_txt = _ordinal(k)
            main = f"the {k_txt} {_WD_FULL[wd]} of {_MONTH_FULL[fuse_month]}"
            hhmm = _fmt_hhmm_for_term(term, default_due_dt)
            if yr_ival > 1:
                main = f"{main} every {yr_ival} years"
            if hhmm:
                main = f"{main} at {hhmm}"
            if bd_filter and any("random day each month" in p for p in m_parts):
                main = f"{main} on a business day"
            # inject pbd/nbd if present
            main = _inject_schedule_suffixes(main, term)
            return main

    # ---------- Interval prefix (w > m > y), plus monthly clarifier logic ----------
    if wk_ival > 1:
        interval_prefix = f"every {wk_ival} weeks: "

    elif mo_ival > 1:
        monthly_prefix = f"every {mo_ival} months"
        clarifier = ""
        if len(monthly_specs) == 1:
            mspec = monthly_specs[0]
            is_nth, pairs = _is_pure_nth_weekday_spec(mspec)
            if is_nth:
                if len(pairs) == 1:
                    k, wd = pairs[0]
                    if k < 0:
                        k_txt = "last" if k == -1 else f"{_ordinal(abs(k))} last"
                    else:
                        k_txt = _ordinal(k)
                    clarifier = f" among months that have the {k_txt} {_WD_FULL[wd]}"
                else:
                    clarifier = " among months that satisfy the specified nth-weekdays"
            else:
                is_dom, doms = _is_pure_dom_spec(mspec)
                if is_dom and any(d >= 29 for d in doms):
                    clarifier = (
                        f" among months that have day {doms[0]}"
                        if len(doms) == 1
                        else " among months that have those days"
                    )

        if clarifier:
            interval_prefix = monthly_prefix + clarifier  # no colon
            suppress_tail = True
        else:
            interval_prefix = monthly_prefix + ": "

    elif yr_ival > 1:
        interval_prefix = f"every {yr_ival} years: "

    # ---------- Build remaining parts (only used when not suppressing tail) ----------
    if w_phrase:
        parts.append(w_phrase)

    if m_parts:
        mp = ", ".join(m_parts)
        if not w_phrase:
            parts.append(mp)
        else:
            parts.append(f"that fall on {mp}")

    if y_parts:
        yp = " or ".join(y_parts) if len(y_parts) > 1 else y_parts[0]
        # If yearly already carries the leading semantics ("one random day ..."),
        # don't prefix with "and within".
        if yp.startswith("one random day"):
            parts.append(yp)
        else:
            if (w_phrase or m_parts):
                parts.append(f"and within {yp}")
            else:
                parts.append(yp)


    if bd_filter and any("random day each month" in p for p in m_parts):
        parts.append("on a business day")

    hhmm = _fmt_hhmm_for_term(term, default_due_dt)

    # ---------- Return early when suppressing tail ----------
    if suppress_tail:
        txt = interval_prefix
        if hhmm:
            txt = f"{txt} at {hhmm}"
        # inject pbd/nbd if present
        return _inject_schedule_suffixes(txt or "any day", term)

    # ---------- Default assembly ----------
    if hhmm:
        parts.append(f"at {hhmm}")
    txt = " ".join(p for p in parts if p)
    if interval_prefix:
        txt = interval_prefix + txt

    # inject pbd/nbd if present
    txt = _inject_prevnext_phrase(txt, term)  # rewrite “prev/next weekday”
    txt = _inject_schedule_suffixes(
        txt or "any day", term
    )  # add @pbd/@nbd/@nw or @bd wording
    return txt or "any day"

def describe_anchor_expr(anchor_expr: str, default_due_dt=None) -> str:
    """
    Natural text for a full anchor expression (OR of AND terms).
    Reuses describe_anchor_term(...) for each AND term and joins them with 'or'.
    """
    if not anchor_expr or not str(anchor_expr).strip():
        return ""
    try:
        dnf = parse_anchor_expr_to_dnf_cached(anchor_expr)
    except Exception:
        return ""

    # Build natural text for each AND term
    nat_terms = []
    for term in dnf:
        try:
            t = describe_anchor_term(term, default_due_dt=default_due_dt)
        except Exception:
            t = ""
        if t:
            nat_terms.append(t)

    if not nat_terms:
        return ""

    # Deduplicate while preserving order
    seen, ordered = set(), []
    for t in nat_terms:
        if t not in seen:
            seen.add(t)
            ordered.append(t)

    # Single vs multiple terms
    return ordered[0] if len(ordered) == 1 else " or ".join(ordered)



def _term_prevnext_wd(term):
    """Return ('next'|'prev', 'Friday') if a prev/next weekday modifier exists, else None."""
    for a in term:
        mods = a.get("mods") or {}
        roll = mods.get("roll")
        if roll in ("next-wd", "prev-wd"):
            wd = mods.get("wd")
            if wd is not None:
                return ("next" if roll == "next-wd" else "prev", _WDNAME.get(wd, ""))
    return None


def _inject_prevnext_phrase(txt: str, term) -> str:
    """
    Prefer rewriting:
      'the last day of each month'   -> 'the previous Friday before the last day of each month'
      'the first day of each month'  -> 'the next Monday after the first day of each month'
    (and the '... of the month/quarter' variants)
    If no known base phrase is found, fall back to ', then the previous/next <Weekday>'.
    """
    tup = _term_prevnext_wd(term)
    if not tup:
        return txt

    dir_word, dayname = tup  # 'prev'|'next', 'Friday'
    rel = "before" if dir_word == "prev" else "after"
    adj = "previous" if dir_word == "prev" else "next"

    # Targets we know how to elegantly rewrite
    targets = [
        "the last day of each month",
        "the first day of each month",
        "the last day of the month",
        "the first day of the month",
        "the last day of each quarter",
        "the first day of each quarter",
    ]

    for target in targets:
        if target in txt:
            pretty = f"the {adj} {dayname} {rel} {target}"
            return txt.replace(target, pretty)

    # Fallback: insert succinctly before any time tail
    phrase = f", then the {adj} {dayname}"
    if " at " in txt:
        head, sep, tail = txt.partition(" at ")
        return f"{head}{phrase} at {tail}"
    return txt + phrase


def describe_anchor_dnf(dnf: list, task: dict) -> str:
    """
    Render the whole expression (OR of AND-terms) into one sentence and append mode.
    First, try special compressions (bucketed monthly rand), else fall back.
    """
    def _mode_tail(mode: str) -> str:
        if mode == "all":
            return "backfill all missed anchors"
        if mode == "flex":
            return "skip past anchors; respect future anchors"
        if mode == "skip":
            return "skip missed anchors"
        return ""

    def _join_terms(terms: list[str]) -> str:
        if not terms:
            return ""
        if len(terms) == 1:
            return terms[0]
        if len(terms) == 2:
            return f"either {terms[0]} or {terms[1]}"
        return "either " + ", ".join(terms[:-1]) + ", or " + terms[-1]

    # Special-case compression
    bucket = _try_bucket_rand_monthly(dnf, task)
    if bucket:
        mode = (task.get("anchor_mode") or "skip").lower()
        tail = _mode_tail(mode)
        return f"{bucket}; {tail}" if tail else bucket

    # Fallback: per-term descriptions OR-joined
    due_dt = parse_dt_any(task.get("due")) if task else None
    terms = [describe_anchor_term(term, due_dt) for term in (dnf or [])]
    if not terms:
        return ""
    sentence = _join_terms(terms)
    mode = (task.get("anchor_mode") or "skip").lower()
    tail = _mode_tail(mode)
    return f"{sentence}; {tail}" if tail else sentence


def _normalize_range_token(tok: str) -> str | None:
    """Return 'A–B' for monthly range tokens like '1..7'; else None."""
    s = (tok or "").strip().lower()
    m = _safe_match(_int_range_re, s)
    if not m:
        return None
    a, b = [int(x) for x in s.split("..")]
    # Keep presentation simple; negatives allowed (already validated upstream)
    return f"{a}–{b}"


def _rand_bucket_signature(term: list[dict]) -> tuple | None:
    """
    For a term shaped like (m:range + m:rand) with optional @t and @bd,
    return a signature tuple for grouping across OR terms:
      (interval, time_str, bd_flag)
    and the normalized single range 'A–B'.
    Returns None if the term doesn't match this pattern exactly.
    """
    has_rand = False
    range_norm = None
    ival_seen = None
    time_str = None
    bd_flag = False

    # Reject weekly or yearly atoms in this special compression
    for a in term:
        typ = (a.get("typ") or a.get("type") or "").lower()
        if typ in ("w", "y"):
            return None

    for a in term:
        typ = (a.get("typ") or a.get("type") or "").lower()
        spec = str(a.get("spec") or a.get("value") or "").lower()
        ival = int(a.get("ival") or a.get("intv") or 1)
        if typ != "m":
            return None
        if spec == "rand":
            has_rand = True
            ival_seen = ival if ival_seen is None else ival_seen
            # time / bd from this atom's mods (or the range atom—must be identical across terms)
            mods = a.get("mods") or {}
            if time_str is None:
                tmod = mods.get("t")
                if isinstance(tmod, tuple):
                    time_str = f"{tmod[0]:02d}:{tmod[1]:02d}"
                elif isinstance(tmod, str) and tmod:
                    time_str = tmod
            bd_flag = bd_flag or bool(mods.get("bd") or (mods.get("wd") is True))
        else:
            # must be exactly one monthly range token like '1..7'
            rn = _normalize_range_token(spec)
            if not rn:
                return None
            if range_norm and rn != range_norm:
                # multiple different ranges inside the same term → not a bucket term
                return None
            range_norm = rn
            ival_seen = ival if ival_seen is None else ival_seen
            mods = a.get("mods") or {}
            if time_str is None:
                tmod = mods.get("t")
                if isinstance(tmod, tuple):
                    time_str = f"{tmod[0]:02d}:{tmod[1]:02d}"
                elif isinstance(tmod, str) and tmod:
                    time_str = tmod
            bd_flag = bd_flag or bool(mods.get("bd") or (mods.get("wd") is True))

    if not (has_rand and range_norm):
        return None

    return (ival_seen or 1, time_str, bd_flag, range_norm)


def _try_bucket_rand_monthly(dnf: list[list[dict]], task: dict) -> str | None:
    """
    If all OR-terms are '(m:A..B + m:rand)' with the same modifiers/interval,
    compress to: 'one random [business] day in each monthly bucket (days A–B, ...)[ at HH:MM]'.
    Returns a sentence or None if not applicable.
    """
    if not dnf or any(len(term) == 0 for term in dnf):
        return None

    sig = None
    ranges = []
    for term in dnf:
        res = _rand_bucket_signature(term)
        if not res:
            return None
        s = (res[0], res[1], res[2])  # (ival, time, bd)
        if sig is None:
            sig = s
        elif s != sig:
            return None
        ranges.append(res[3])

    # Sort ranges by their numeric start to make the prose nice
    def _start_val(r):
        a = r.split("–", 1)[0]
        try:
            return int(a)
        except:
            return 0

    ranges = sorted(ranges, key=_start_val)

    ival, time_str, bd_flag = sig
    parts = []
    lead = "one random "
    if bd_flag:
        lead += "business "
    lead += "day"
    if ival and int(ival) > 1:
        lead = f"every {ival} months: " + lead
    parts.append(lead + " in each monthly bucket")
    # Join buckets compactly
    buckets = ", ".join([f"days {r}" for r in ranges])
    parts.append(f"({buckets})")
    if time_str:
        parts.append(f"at {time_str}")
    return " ".join(parts)


# -------- ----------


def _split_inline_items_respecting_t_lists(s: str) -> list[str]:
    """Split comma-list items, but keep commas inside '@t=HH:MM,HH:MM' values."""
    if not s:
        return []
    out = []
    buf = []
    in_t_value = False
    i, n = 0, len(s)

    def flush():
        tok = "".join(buf).strip()
        if tok:
            out.append(tok)
        buf.clear()

    while i < n:
        ch = s[i]

        if ch == "@":
            if s[i:i+3].lower() == "@t=":
                in_t_value = True
            else:
                in_t_value = False
            buf.append(ch)
            i += 1
            continue

        if ch == ",":
            if in_t_value:
                # If the comma separates times inside @t=..., keep it.
                # Heuristic: if the next token looks like a new list item (has '@' or starts with alpha / '-' / '(' / '|' / '&'),
                # treat comma as an item separator; otherwise treat it as part of the @t list (even if the token is invalid,
                # so @t validation can emit the correct error).
                j = i + 1
                while j < n and s[j].isspace():
                    j += 1
                k = j
                while k < n and s[k] != ",":
                    k += 1
                nxt = s[j:k].strip()
                if nxt and ("@" not in nxt) and (not nxt[0].isalpha()) and (nxt[0] not in "-(|&"):
                    buf.append(ch)
                    i += 1
                    continue
                # Otherwise comma ends this item
                flush()
                in_t_value = False
                i += 1
                continue

            # Normal separator (not inside @t list)
            flush()
            i += 1
            continue

        buf.append(ch)
        i += 1

    flush()
    return out


def _parse_group_with_inline_mods(typ: str, ival: int, spec: str, outer_mods_str: str):
    """
    If 'spec' is a comma-list where any item contains an inline '@...' modifier,
    rewrite it into a DNF OR list:  [ [atom1], [atom2], ... ]  (each item its own atom).
    Returns None if no rewrite is needed.

    Critical disambiguation:
      - If only the LAST item contains '@...', treat it as group-level mods
        (e.g. 'w:mon,tue,wed@t=09:00,12:00'), so we return None and let the normal
        'spec/mods' split handle it.
    """
    toks = [t.strip() for t in _split_inline_items_respecting_t_lists(str(spec or "")) if t.strip()]
    if len(toks) < 2 or not any("@" in t for t in toks):
        return None  # nothing to do

    if outer_mods_str.strip():
        raise ParseError(
            "Cannot mix group-level modifiers (after ':') with per-item modifiers in the same list. "
            "Choose one style: either 'w:mon@t=09:00,fri@t=15:00' (per-item) "
            "or 'w:mon,fri@t=09:00,15:00' (group)."
        )

    at_idxs = [i for i, t in enumerate(toks) if "@" in t]
    if len(at_idxs) == 1 and at_idxs[0] == (len(toks) - 1):
        return None

    # Build OR-of-singletons DNF
    or_terms = []
    for tok in toks:
        if "@" in tok:
            item_spec, item_mods_str = tok.split("@", 1)
        else:
            item_spec, item_mods_str = tok, ""
        item_spec = item_spec.strip().lower()
        item_mods = _parse_atom_mods(item_mods_str.strip())
        # Inline per-item modifier lists are intended only for per-item time selection.
        # To keep the grammar resilient and avoid ambiguous "trailing mods", we disallow
        # any non-time modifiers inside an inline item (e.g. '@bd', '@next-mon', '@+1d').
        if item_mods_str.strip():
            if item_mods.get("roll") or item_mods.get("wd") is not None or item_mods.get("bd") or (item_mods.get("day_offset") or 0) != 0:
                raise ParseError(
                    "Inline per-item modifiers in comma-lists only support '@t=HH:MM[,HH:MM...]'. "
                    "For other modifiers (e.g. '@bd'), use group style like 'w:mon,tue@bd@t=09:00,12:00' "
                    "or explicit OR terms with '|', e.g. '(w:mon@t=09:00) | (w:tue@bd@t=12:00)'."
                )
        or_terms.append([{"typ": typ, "spec": item_spec, "ival": ival, "mods": item_mods}])

    return or_terms


# ------------------------------------------------------------------------------
# Date/time parsing & humanization
# ------------------------------------------------------------------------------
def coerce_int(v, default=None):
    """Safely convert value to int, handling floats and strings."""
    try:
        if v is None:
            return default
        if isinstance(v, bool):
            return default
        if isinstance(v, int):
            return v if abs(v) <= (2**63 - 1) else default
        if isinstance(v, float):
            if not math.isfinite(v):
                return default
            iv = int(round(v))
            return iv if abs(iv) <= (2**63 - 1) else default
        s = str(v).strip()
        if _int_floatish_re.fullmatch(s):
            iv = int(float(s))
            return iv if abs(iv) <= (2**63 - 1) else default
        iv = int(s)
        return iv if abs(iv) <= (2**63 - 1) else default
    except Exception:
        return default


def parse_dt_any(s: str):
    """Parse datetime from string using multiple formats."""
    if not s:
        return None
    s = str(s)
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except Exception:
            pass
    try:
        d = datetime.strptime(s[:10], "%Y-%m-%d")
        return d.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def month_len(y, m):
    """Get number of days in month."""
    import calendar

    return calendar.monthrange(y, m)[1]


def add_months(d: date, months: int) -> date:
    """Add months to date, handling month-end correctly."""
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    last = month_len(y, m)
    return date(y, m, min(d.day, last))


def months_days_between(d1: date, d2: date):
    """Calculate months and days between two dates."""
    sign = 1
    if d2 < d1:
        d1, d2 = d2, d1
        sign = -1
    months = (d2.year - d1.year) * 12 + (d2.month - d1.month)
    if add_months(d1, months) > d2:
        months -= 1
    anchor = add_months(d1, months)
    days = (d2 - anchor).days
    return sign * months, sign * days


def humanize_delta(from_dt: datetime, to_dt: datetime, use_months_days: bool):
    """Human-readable time difference between datetimes."""
    td = to_dt - from_dt
    if use_months_days:
        m, d = months_days_between(from_dt.date(), to_dt.date())
        future = m > 0 or (m == 0 and d > 0)
        label = "in" if future else "overdue by"
        m, d = abs(m), abs(d)
        parts = []
        if m:
            parts.append(f"{m}mo")
        if d or not parts:
            parts.append(f"{d}d")
        return f"{label} " + " ".join(parts)
    secs = int(abs(td.total_seconds()))
    label = "in" if td.total_seconds() > 0 else "overdue by"
    days, rem = divmod(secs, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)
    if days:
        return f"{label} {days}d {hours}h"
    if hours:
        return f"{label} {hours}h {minutes}m"
    return f"{label} {minutes}m"


def _active_mod_keys(mods: dict) -> set:
    """Return only modifiers that are actually 'used' (truthy / non-zero)."""
    act = set()
    for k, v in (mods or {}).items():
        if v in (None, False, 0, 0.0, "", []):  # all 'inactive' values
            continue
        act.add(k)
    return act


# --- Atom helpers (robust field access) ---
def _atype(atom):
    return (atom.get("typ") or atom.get("type") or "").lower()


def _aspec(atom):
    return str(atom.get("spec") or atom.get("value") or "").lower()


def _amods(atom):
    return atom.get("mods") or {}


def _ainterval(atom):  # monthly /N (accept both ival and intv)
    try:
        return int(atom.get("ival") or atom.get("intv") or 1)
    except Exception:
        return 1


# --- Random anchors helpers ---------------------------------------------------

_WD = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}


def _week_monday(d: date) -> date:
    return d - timedelta(days=d.weekday())  # Monday of ISO week


def _seeded_int(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:4], "big")


def _weekly_rand_pick(iso_year: int, iso_week: int, mods: dict) -> int:
    """Return a deterministic weekday 0..6 for (year, week). @bd limits pool to Mon–Fri."""
    pool = (
        [0, 1, 2, 3, 4]
        if (mods.get("bd") or mods.get("wd") is True)
        else [0, 1, 2, 3, 4, 5, 6]
    )
    key = f"{WRAND_SALT}|{iso_year}|{iso_week}|{'bd' if len(pool)==5 else 'all'}"
    n = _seeded_int(key)
    return pool[n % len(pool)]


def _days_in_month(y, m):
    """Get number of days in month (optimized)."""
    if m == 12:
        return 31
    return (date(y, m + 1, 1) - timedelta(days=1)).day


def _is_bd(dt: _date):  # business day
    return dt.weekday() < 5


def _sha_pick(seq_len: int, seed_key: str) -> int:
    """Deterministic random selection using SHA-256."""
    h = hashlib.sha256(seed_key.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % seq_len


def _term_rand_info(term):
    for i, a in enumerate(term):
        typ = (a.get("typ") or a.get("type") or "").lower()
        spec = str(a.get("spec") or a.get("value") or "").lower()
        if typ == "m" and spec == "rand":
            return ("m", {"mods": a.get("mods") or {}, "ival": int(a.get("ival") or a.get("intv") or 1), "atom_idx": i})
        if typ == "y":
            if spec == "rand":
                return ("y", {"mods": a.get("mods") or {}, "month": None, "atom_idx": i})
            if spec.startswith("rand-"):
                mm = int(spec.split("-", 1)[1])
                return ("y", {"mods": a.get("mods") or {}, "month": mm, "atom_idx": i})
    return (None, None)



def _filter_by_w(dt_list: list[_date], term: list[dict]):
    """Filter dates by weekday constraints in term."""
    allowed = None
    for a in term:
        if _atype(a) != "w":
            continue
        spec = (_aspec(a) or "").lower()
        wset = _weekly_spec_to_wset(spec, mods=a.get("mods") or {})
        allowed = wset if allowed is None else (allowed & wset)
    if allowed is None:
        return dt_list
    if not allowed:
        return []
    return [d for d in dt_list if d.weekday() in allowed]


@_ttl_lru_cache(maxsize=128)
def _month_tokens_for_atom_cached(y: int, m: int, spec: str) -> set[int]:
    """
    Cached version of month token expansion.
    For a monthly atom, return set of day numbers in (y,m) that match the spec.
    """
    spec = _expand_monthly_aliases(spec)
    ndays = _days_in_month(y, m)
    out = set()

    # business-day like '5bd' or '-2bd'
    m2 = _bd_re.match(spec)
    if m2:
        k = int(m2.group(1))
        # build list of business days
        bds = [d for d in range(1, ndays + 1) if _date(y, m, d).weekday() < 5]
        if not bds:
            return out
        if k > 0:
            if k <= len(bds):
                out.add(bds[k - 1])
        else:
            k = -k
            if k <= len(bds):
                out.add(bds[-k])
        return out

    # nth-weekday: '2mon', 'last-fri', '-1fri'
    m3 = _nth_weekday_re.match(spec)
    if m3:
        idx, wd = m3.group(1), _WD[m3.group(2)]
        # all days of that weekday in month
        days = [d for d in range(1, ndays + 1) if _date(y, m, d).weekday() == wd]
        if not days:
            return out
        if idx == "last":
            out.add(days[-1])
            return out
        k = int(re.sub(r"(st|nd|rd|th)$", "", idx))
        if k > 0 and k <= len(days):
            out.add(days[k - 1])
        elif k < 0 and -k <= len(days):
            out.add(days[k])
        return out

    # range A..B (A or B may be negative)
    if ".." in spec:
        a_s, b_s = spec.split("..", 1)
        try:
            a_i = int(a_s)
            b_i = int(b_s)
        except:
            return out

        def norm(n):
            return ndays + n + 1 if n < 0 else n

        lo, hi = norm(a_i), norm(b_i)
        lo = max(1, lo)
        hi = min(ndays, hi)
        if lo <= hi:
            out.update(range(lo, hi + 1))
        return out

    # single int (may be negative)
    try:
        k = int(spec)
        if k < 0:
            k = _days_in_month(y, m) + k + 1
        if 1 <= k <= ndays:
            out.add(k)
    except:
        pass
    return out


def _month_tokens_for_atom(a: dict, y: int, m: int) -> set[int]:
    """Wrapper for cached month token expansion."""
    spec = str(a.get("spec")).lower().strip()
    return _month_tokens_for_atom_cached(y, m, spec)


def _term_candidates_in_month(
    term: list[dict], y: int, m: int, rand_atom_idx: int, bd_only: bool
):
    """
    Build candidate dates in (y,m) that satisfy all *other* atoms in 'term',
    ignoring the rand atom itself.
    """
    days = list(range(1, _days_in_month(y, m) + 1))
    dates = [_date(y, m, d) for d in days]

    # filter business days if requested on the rand atom
    if bd_only:
        dates = [d for d in dates if _is_bd(d)]

    # apply w: filters (weekdays)
    dates = _filter_by_w(dates, term)

    # apply other monthly tokens in this term
    msets = []
    for idx, a in enumerate(term):
        if idx == rand_atom_idx:
            continue
        if _atype(a) != "m":
            continue
        sp = _aspec(a)
        if sp == "rand":  # already handled
            continue
        msets.append(_month_tokens_for_atom(a, y, m))
    if msets:
        allowed_days = set.intersection(*msets) if msets else set()
        dates = [d for d in dates if d.day in allowed_days]

    # Yearly gating (e.g., y:04-20..05-15) — keep only DOMs allowed by the y: windows
    y_specs = [str(a.get("spec") or "") for a in term if _atype(a) == "y"]
    if y_specs:
        allowed_dom = _doms_allowed_by_year(y, m, y_specs)
        if allowed_dom:
            dates = [d for d in dates if d.day in allowed_dom]
        else:
            dates = []

    return dates


def _months_since(seed_local: _date, y: int, m: int) -> int:
    """Calculate months between seed date and given year/month."""
    return (y - seed_local.year) * 12 + (m - seed_local.month)


# -------- cp duration ----------
def parse_cp_duration(dur: str):
    """Parse ISO 8601 duration string to timedelta."""
    if not dur:
        return None
    m = _cp_re.match(dur.strip())
    if not m:
        return None
    return timedelta(
        weeks=int(m.group("w") or 0),
        days=int(m.group("d") or 0),
        hours=int(m.group("h") or 0),
        minutes=int(m.group("m") or 0),
        seconds=int(m.group("s") or 0),
    )


# -------- Anchor parser (DNF with mods) ----------
class ParseError(Exception):
    pass


def _parse_hhmm(s: str):
    """Parse HH:MM time string."""
    m = _hhmm_re.match(s)
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)))


def _parse_atom_head(head: str) -> tuple[str, int]:
    """
    Parse the atom head:  'w' | 'm' | 'y'  with optional '/N' (1..100).
    Examples: 'w', 'w/2', 'm', 'm/3', 'y/4'
    Returns (typ, ival).
    """
    h = (head or "").strip().lower()
    m = re.fullmatch(r'(w|m|y)(?:/(\d{1,3}))?$', h)
    if not m:
        raise ParseError(
            f"Invalid anchor head '{head}'. Expected 'w', 'm', or 'y' with optional '/N', "
            "e.g., 'w/2', 'm/3', 'y/4'."
        )
    typ = m.group(1)
    ival = int(m.group(2) or 1)
    if ival < 1:
        ival = 1
    if ival > 100:
        ival = 100
    return typ, ival




def _parse_atom_mods(mods_str: str):
    """Parse atom modifiers (e.g., @t=09:00 or @t=09:00,12:00,18:00, @+1d)."""
    mods = {"t": None, "roll": None, "wd": None, "bd": False, "day_offset": 0}
    if not mods_str:
        return mods

    def _parse_time_list(v: str):
        parts = _split_csv_tokens(v)
        if not parts:
            return None
        out = []
        seen = set()
        for p in parts:
            hhmm = _parse_hhmm(p)
            if not hhmm:
                raise ParseError(f"Invalid time in @t=HH:MM[,HH:MM...]: '{p}'")
            if hhmm not in seen:
                out.append(hhmm)
                seen.add(hhmm)
        return out

    for raw in mods_str.split("@"):
        tok = raw.strip().lower()
        if not tok:
            continue

        if tok in ("nw", "pbd", "nbd"):
            mods["roll"] = tok
            continue

        if tok == "bd":
            # business day filter (weekdays only) — distinct from wd=int
            mods["bd"] = True
            continue

        m = _next_prev_wd_re.match(tok)
        if m:
            mods["roll"] = f"{m.group(1)}-wd"
            mods["wd"] = _WEEKDAYS[m.group(2)]  # 0..6 target weekday
            continue

        # Time modifier: support a single @t=... per atom, with comma-separated HH:MM list.
        if tok.startswith("t="):
            if mods["t"] is not None:
                raise ParseError(
                    "Duplicate '@t=' modifier. Use a single '@t=HH:MM,HH:MM,...' list."
                )
            tval = tok.split("=", 1)[1].strip()
            tlist = _parse_time_list(tval)
            if not tlist:
                raise ParseError(f"Invalid time in @t=HH:MM[,HH:MM...]: '{tok}'")
            mods["t"] = tlist[0] if len(tlist) == 1 else tlist
            continue

        m = _day_offset_re.match(tok)
        if m:
            mods["day_offset"] += int(m.group(1))
            continue

        raise ParseError(f"Unknown modifier '@{tok}'")
    return mods


@_ttl_lru_cache(maxsize=512)
def _parse_y_token_cached(tok: str, fmt: str):
    """Parse yearly token (e.g., '15-02' or 'q1')."""
    tok = tok.strip().lower()
    if tok in _QUARTERS:
        return ("quarter", tok)
    m = re.fullmatch(r"q([1-4])([sme])", tok)
    if m:
        return ("quarter", tok)
    m = _y_token_re.match(tok)
    if not m:
        return None
    a, b = m.group(1), m.group(2)
    if b.isalpha():
        if b not in _MONTHS:
            return None
        b = _MONTHS[b]
    else:
        b = int(b)
    a = int(a)
    if fmt == "DM":
        d, mn = a, b
    else:
        mn, d = a, b
    if not (1 <= mn <= 12):
        return None
    if mn in (4, 6, 9, 11) and d == 31:
        return None
    if mn == 2 and d > 29:
        return None
    if not (1 <= d <= 31):
        return None
    return ("day", (mn, d))


def _parse_y_token(tok: str):
    return _parse_y_token_cached(tok, _yearfmt())


# ------------------------------------------------------------------------------
# Anchor DNF parser
# ------------------------------------------------------------------------------
def parse_anchor_expr_to_dnf(s: str):
    """Parse anchor expression into Disjunctive Normal Form."""
    # Accept anchors wrapped in quotes from the CLI; normalize rand-month alias.
    s = _unwrap_quotes(s or "").strip()
    if len(s) > 1024:
        raise ParseError("Anchor expression too long (max 1024 characters).")
    s = re.sub(r'\b(\d{2})-rand\b', r'rand-\1', s)
    s = _rewrite_weekly_multi_time_atoms(s)
    i = 0
    n = len(s)

    # --- FATAL: numeric yearly tokens using ':' (should be '-') ----------------
    # Scan for every 'y[:spec]' tail; ignore modifiers after '@'; then check
    # each comma-separated token for a numeric colon form like '05:15' or
    # '05:01:06:30'. Month-name forms must use '..' for ranges.
    def _fatal_bad_colon_in_year_tail(tail: str) -> str | None:
        head = tail.split("@", 1)[0]  # strip modifiers
        for tok in _split_csv_tokens(head):
            # numeric with ':'  → fatal (e.g., '05:15' or '05:01:06:30')
            if re.fullmatch(r"\d{2}:\d{2}(?::\d{2}:\d{2})?", tok):
                fmt = (globals().get("ANCHOR_YEAR_FMT") or "DM").upper()
                example = "06-01" if fmt == "MD" else "01-06"
                return (f"Yearly token '{tok}' uses ':' between numbers. "
                        f"Use '-' and order per ANCHOR_YEAR_FMT={fmt}. Example: '{example}'.")
            if ":" in tok:
                return "Yearly ranges must use '..' (e.g., '01-01..12-31', 'q1..q2')."
        return None

    # Find every 'y' atom head (with optional '/N') then grab its tail up to +,|,) or EOL
    for m in re.finditer(r"\by\s*(?:/\d+)?\s*:", s):
        j = m.end()  # start of tail
        k = j
        while k < len(s) and s[k] not in "+|)":
            k += 1
        tail = s[j:k]
        fatal_msg = _fatal_bad_colon_in_year_tail(tail)
        if fatal_msg:
            # FIX: Raise ParseError instead of returning tuple
            raise ParseError(fatal_msg)


    def skip_ws():
        nonlocal i
        while i < n and s[i].isspace():
            i += 1

    def take(ch):
        nonlocal i
        if i < n and s[i] == ch:
            i += 1
            return True
        return False

    def parse_atom():
        """
        Parse a single anchor atom and return it in DNF as a list of terms, where each
        term is a list of atom dicts:
        return value shape: [ [ {typ, spec, ival, mods}, ... ], ... ]

        Expectations (provided by the outer parser scope):
        - nonlocal i, n, s
        - helpers: skip_ws(), take(ch), ParseError,
                    _parse_atom_head(head) -> (typ, ival),
                    _parse_atom_mods(mods_str) -> dict,
                    _parse_group_with_inline_mods(typ, ival, tail, leading_mods_str) -> dnf|None
        """
        nonlocal i
        skip_ws()

        # ---- read head like "m", "w/2", "y", "m/3" etc. (up to the ':') ----
        start = i
        while i < n and s[i] not in ":()+|":
            i += 1
        head = s[start:i].strip()

        # require ':' after head (type + optional /N)
        if not take(":"):
            raise ParseError("Expected ':' after anchor type. Example 'w:mon', 'm:-1', 'y:06-01'")

        # ---- read the tail (spec + optional '@mods') until atom terminator ----
        # stop at ')' or '|' or '+' (but allow '@+Nd' inside modifiers)
        start = i
        while i < n:
            ch = s[i]
            if ch in ")|":
                break
            if ch == "+" and not (i > start and s[i - 1] == "@"):
                break
            i += 1

        full_tail = s[start:i].strip()  # e.g. "mon,tue@t=09:00" or "1st-mon@t=08:00,fri@t=15:00"

        # ─────────────────────────────────────────────────────────────
        # Guards against comma "joining" separate atoms:
        #  - legal: w:mon,tue            (list inside spec)
        #           m:1st-mon,3rd-wed    (list inside spec)
        #  - illegal: m:31@t=14:00,w:sun (comma used to start a new anchor)
        if re.search(r"@[^)]*?,\s*(?:w|m|y)(?:/|:)", full_tail):
            raise ParseError(
                "It looks like you used a comma to join anchors. "
                "Use '+' (AND) or '|' (OR), e.g. 'm:31@t=14:00 | w:sun@t=22:00'."
            )

        if re.search(r",\s*(?:w|m|y)(?:/|:)", full_tail):
            raise ParseError(
                "Anchors must be joined with '+' (AND) or '|' (OR). "
                "For example: 'm:31 + w:sun' or 'm:31 | w:sun'."
            )
        # ─────────────────────────────────────────────────────────────

        # Parse type + optional /N from head
        typ, ival = _parse_atom_head(head)
        tlo = (typ or "").lower()

        # Try inline per-item mods rewriter BEFORE we split spec/mods.
        # This handles cases like "w:mon@t=09:00,fri@t=15:00" (→ OR of per-item atoms).
        dnf_or = _parse_group_with_inline_mods(tlo, ival, full_tail, "")
        if dnf_or is not None:
            return dnf_or  # already an OR over singletons with their own mods

        # Split into spec and mods (if any)
        spec, mods_str = (full_tail.split("@", 1) + [""])[:2]

        # Normalize monthly ordinal-hyphen tokens (accept "1st-mon", "2nd-tue", ...)
        # Your monthly engine already understands the compact "1mon" form, so normalize to that.
        if tlo == "m":
            def _ord_norm(mo: re.Match) -> str:
                return f"{mo.group(1)}{mo.group(3).lower()}"
            spec = re.sub(
                r'\b([1-5])\s*(st|nd|rd|th)\s*-\s*(mon|tue|wed|thu|fri|sat|sun)\b',
                _ord_norm,
                spec,
                flags=re.IGNORECASE
            )

        # Special-case: w:rand,mon → OR of singletons so 'rand' doesn't mingle with fixed dows
        if tlo == "w":
            toks = _split_csv_lower(spec)
            if "rand" in toks and len(toks) > 1:
                mods = _parse_atom_mods(mods_str)
                return [
                    [
                        {
                            "typ": "w",
                            "spec": t,
                            "ival": ival,
                            "mods": mods,
                        }
                    ]
                    for t in toks
                ]

        mods = _parse_atom_mods(mods_str)

        # Return a single-term, single-atom DNF node
        return [[{"typ": tlo, "spec": spec.strip().lower(), "ival": ival, "mods": mods}]]


    def parse_factor(depth: int = 0):
        if depth > 50:
            raise ParseError("Expression nesting too deep")
        skip_ws()
        if take("("):
            res = parse_expr(depth + 1)
            skip_ws()
            if not take(")"):
                raise ParseError("Unclosed '('")
            return res
        return parse_atom()

    def and_merge(A, B):
        return [ta + tb for ta in A for tb in B]

    def parse_term(depth: int = 0):
        nonlocal i
        left = parse_factor(depth)
        while True:
            pos = i
            skip_ws()
            if not take("+"):
                i = pos
                break
            right = parse_factor(depth)
            left = and_merge(left, right)
        return left

    def parse_expr(depth: int = 0):
        nonlocal i
        left = parse_term(depth)
        while True:
            pos = i
            skip_ws()
            if not take("|"):
                i = pos
                break
            right = parse_term(depth)
            left = left + right
        return left

    res = parse_expr(0); skip_ws()
    if i != n:
        raise ParseError("Unexpected trailing characters")
    dnf = _rewrite_quarters_in_context(res)
    dnf = _rewrite_year_month_aliases_in_context(dnf)   # NEW
    _validate_year_tokens_in_dnf(dnf)
    _validate_and_terms_satisfiable(dnf, ref_d=date.today())
    return dnf


@_ttl_lru_cache(maxsize=256)
def _parse_anchor_expr_to_dnf_cached_obj(s: str, fmt: str):
    return parse_anchor_expr_to_dnf(s)


def parse_anchor_expr_to_dnf_cached(s: str):
    """Cached parse returning a fresh object (avoid shared mutable structures)."""
    if not s:
        return []
    key = _unwrap_quotes(s or "").strip()
    if not key:
        return []
    res = copy.deepcopy(_parse_anchor_expr_to_dnf_cached_obj(key, _yearfmt()))
    _emit_cache_metrics()
    if os.environ.get("NAUTICAL_CLEAR_CACHES") == "1":
        _clear_all_caches()
    return res



# ------------------------------------------------------------------------------
# Anchor validators
# ------------------------------------------------------------------------------
class YearTokenFormatError(ParseError):
    pass


def _validate_yearly_token_format(spec: str):
    """
    Enforce yearly numeric format and known allowances.
    Fatal on numeric tokens that use ':' instead of '-'.
    """
    fmt = (globals().get("ANCHOR_YEAR_FMT") or "DM").upper()
    if not spec:
        return

    tokens = _split_csv_lower(spec)

    for tok in tokens:
        s = tok

        # Allow 'rand' and 'rand-XX'
        if s == "rand" or re.fullmatch(r"rand-\d{2}", s):
            continue

        # FATAL: numeric with ':' (e.g., '05:15' or '05:01:06:30')
        if re.fullmatch(r"\d{2}:\d{2}(?::\d{2}:\d{2})?", s):
            example = "06-01" if fmt == "MD" else "01-06"
            raise YearTokenFormatError(
                f"Yearly token '{tok}' uses ':' between numbers. "
                f"Use '-' and order per ANCHOR_YEAR_FMT={fmt}. Example: '{example}'."
            )

        # Accept standard numeric day-month (single or range) — parser ensures order (MD/DM)
        if re.fullmatch(r"\d{2}-\d{2}(?:\.\.\d{2}-\d{2})?", s):
            continue

        # Accept month aliases and month..month; rewritten downstream
        if re.fullmatch(r"(?:[a-z]{3}|\d{2})\.\.(?:[a-z]{3}|\d{2})", s):
            continue
        if re.fullmatch(r"[a-z]{3}", s):  # 'apr', 'jul'
            continue

        # Accept quarters (rewritten earlier)
        if re.fullmatch(r"q[1-4](?:\.\.q[1-4])?", s):
            continue

        # Everything else is invalid
        raise YearTokenFormatError(f"Unknown yearly token '{tok}'. Expected day-month, month alias, or quarter.")
    bad = None

    def _pair(a:int, b:int) -> tuple[int,int]:  # returns (day, month)
        return (b, a) if fmt == "MD" else (a, b)

    def _check(mm:int, dd:int) -> str | None:
        if not (1 <= mm <= 12):
            return f"month '{mm:02d}' is invalid"
        if not (1 <= dd <= 31):
            return f"day '{dd:02d}' is invalid"
        return None

    for tok in tokens:
        s = tok.strip().lower()

        if s == "rand":
            continue
        m_randm = re.fullmatch(r"rand-(\d{2})", s)
        if m_randm:
            mm = int(m_randm.group(1))
            if 1 <= mm <= 12:
                continue
            raise YearTokenFormatError(f"Invalid month in yearly token '{tok}'. Expected 01..12.")


        # Proper numeric tokens: DD-MM or MM-DD, with optional range tail (V2 '..')
        m = re.fullmatch(r"(\d{2})-(\d{2})(?:\.\.(\d{2})-(\d{2}))?$", s)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            d1, m1 = _pair(a, b)
            err = _check(m1, d1)
            if err:
                bad = (tok, err); break
            if m.group(3):
                c, d = int(m.group(3)), int(m.group(4))
                d2, m2 = _pair(c, d)
                err2 = _check(m2, d2)
                if err2:
                    bad = (tok, err2); break
                if (m2, d2) < (m1, d1):
                    bad = (tok, "end precedes start"); break
            continue

        # NEW: colon-only separators like '05:15' or '05:15:06:20' → friendly error
        m_col1 = re.fullmatch(r"(\d{2}):(\d{2})$", s)
        m_col2 = re.fullmatch(r"(\d{2}):(\d{2}):(\d{2}):(\d{2})$", s)
        if m_col1 or m_col2:
            if m_col1:
                A, B = int(m_col1.group(1)), int(m_col1.group(2))
                ex = f"{A:02d}-{B:02d}" if fmt == "MD" else f"{B:02d}-{A:02d}"
            else:
                A, B, C, D = map(int, m_col2.groups())
                ex = (f"{A:02d}-{B:02d}..{C:02d}-{D:02d}" if fmt == "MD"
                      else f"{B:02d}-{A:02d}..{D:02d}-{C:02d}")
            raise YearTokenFormatError(
                f"Yearly token '{tok}' uses ':' between numbers. "
                f"Use '-' and order per ANCHOR_YEAR_FMT={fmt}. Example: '{ex}'."
            )

        if ":" in s:
            raise YearTokenFormatError(
                f"Yearly ranges must use '..' (e.g., '01-01..12-31', 'q1..q2')."
            )

        # If it looks numeric-ish but didn’t match the proper pattern, nudge with a general hint
        if any(ch.isdigit() for ch in s) and any(ch in s for ch in "-:"):
            ex = "MM-DD" if fmt == "MD" else "DD-MM"
            raise YearTokenFormatError(
                f"Yearly token '{tok}' doesn’t match ANCHOR_YEAR_FMT={fmt}. "
                f"Expected {ex} or {ex}..{ex}."
            )
        # Non-numeric tokens (e.g., month names/quarters) are rewritten earlier and can pass here

    if bad:
        tok, reason = bad
        sug = ("Did you mean MM-DD? e.g., '04-20'."
               if fmt == "MD" else "Did you mean DD-MM? e.g., '20-04'.")
        raise YearTokenFormatError(
            f"Yearly token '{tok}' doesn’t match ANCHOR_YEAR_FMT={fmt}. {reason}. {sug}"
        )


def _validate_year_tokens_in_dnf(dnf):
    for term in dnf:
        for a in term:
            if (a.get("typ") or "").lower() == "y":
                spec = (a.get("spec") or "").strip()
                _validate_yearly_token_format(spec)


# ---- AND-term satisfiability guard -----------------------------------------
class AndTermUnsatisfiable(ParseError):
    pass


_LEAP_YEAR_FOR_CHECKS = 2028


def _weekday_set_from_weekly_atom(a) -> set[int]:
    """Return {0..6} weekday set for a 'w' atom.

    Handles canonical V2 ranges ('..').
    """
    if (a.get("typ") or "").lower() != "w":
        return set()
    spec = (a.get("spec") or "")
    mods = a.get("mods") or {}
    return _weekly_spec_to_wset(spec, mods=mods)


def _md_pairs_from_yearly_spec(spec: str) -> set[tuple[int, int]]:
    """Expand a yearly spec in a leap year and return {(month, day),...}."""
    if not spec:
        return set()
    try:
        dates = expand_yearly_cached(spec, _LEAP_YEAR_FOR_CHECKS)
    except Exception:
        return set()
    return {(d.month, d.day) for d in dates}


def _quick_weekly_and_check(term: list[dict]) -> None:
    """w + w: if weekday intersection is empty, fail immediately."""
    w_sets = [
        _weekday_set_from_weekly_atom(a)
        for a in term
        if (a.get("typ") or "").lower() == "w"
    ]
    if len(w_sets) >= 2:
        inter = set.intersection(*w_sets) if all(w_sets) else set()
        if not inter:
            raise AndTermUnsatisfiable(
                "Weekly anchors joined with '+' never coincide (e.g., Saturday AND Monday). "
                "Use ',' (OR) or '|' instead."
            )


def _quick_yearly_and_check(term: list[dict]) -> None:
    """y + y: if per-year date set intersection is empty, fail."""
    y_atoms = [a for a in term if (a.get("typ") or "").lower() == "y"]
    if len(y_atoms) < 2:
        return
    md_sets = []
    for ya in y_atoms:
        spec = (ya.get("spec") or "").strip().lower()
        s = _md_pairs_from_yearly_spec(spec)
        if s:
            md_sets.append(s)
    if len(md_sets) >= 2:
        inter = set.intersection(*md_sets)
        if not inter:
            # Build a hint using the normalized specs we just saw
            joined = ", ".join((ya.get("spec") or "").strip().lower() for ya in y_atoms)
            raise AndTermUnsatisfiable(
                "Yearly anchors joined with '+' never overlap within a year. "
                f"If you intended 'either/or', join them with commas: y:{joined}"
            )


def _term_has_any_match_within(
    term: list[dict], start: date, seed: date, years: int = 8
) -> bool:
    """
    If any atom uses 'rand' (weekly or monthly), treat that atom as 'can match this bucket'
    for satisfiability purposes. Actual random is resolved in the scheduler, not here.
    """

    def _matches_or_flexible(a, d):
        typ = (a.get("typ") or "").lower()
        spec = (a.get("spec") or "").lower()
        if typ in ("w", "m") and "rand" in spec:
            return True  # flexible for the validator
        return atom_matches_on(a, d, seed)

    limit = start + timedelta(days=366 * max(1, years))
    d = start
    while d <= limit:
        if all(_matches_or_flexible(a, d) for a in term):
            return True
        d += timedelta(days=1)
    return False


# ------------------------------------------------------------------------------
# Anchor satisfiability checks
# ------------------------------------------------------------------------------
def _validate_and_terms_satisfiable(dnf: list[list[dict]], ref_d: date):
    """
    For each AND-term (inside a DNF expression), ensure it's satisfiable.
    - Fast rejections for weekly+weekly and yearly+yearly.
    - Otherwise, bounded look-ahead scan (8y) using atom_matches_on.
    """
    seed = (
        ref_d  # use now as seed for gating; scheduler will re-evaluate with chain seed
    )
    for term in dnf:
        if len(term) < 2:
            continue  # single-atom terms are trivially satisfiable

        # Fast, structure-aware checks
        _quick_weekly_and_check(term)
        _quick_yearly_and_check(term)

        # Bounded scan for everything else (m+m, m+y, mixes, etc.)
        if not _term_has_any_match_within(term, ref_d, seed, years=8):
            # Build a friendly suggestion by showing a comma-joined alternative
            # like: y:01-03,15-08  or  w:sat,mon  or  m:1,5
            pieces = []
            for a in term:
                typ = (a.get("typ") or "").lower()
                spec = (a.get("spec") or "").strip()
                # For user-facing examples, prefer canonical V2 delimiters where possible.
                if typ in ("w", "m"):
                    try:
                        spec = _normalize_spec_for_acf(typ, spec) or spec
                    except Exception:
                        pass
                if typ:
                    pieces.append(f"{typ}:{spec}" if spec else typ)
            hint = ", ".join(pieces)
            raise AndTermUnsatisfiable(
                "These anchors joined with '+' don't share any possible date. "
                "If you meant 'either/or', join them with ',' (OR) or use '|'. "
                f"Example: {hint.replace(' + ', ', ')}"
            )


# ===== Strict validators (raise ParseError) ==================================


def _validate_weekly_spec(spec: str):
    """Validate weekly specification tokens.

    Canonical (V2) range syntax uses '..' (e.g., w:mon..fri).
    """
    spec = _expand_weekly_aliases(spec)
    toks = _split_csv_lower(spec)
    if not toks:
        raise ParseError(
            f"Weekly spec is empty. Examples: '{_CANON_WEEKLY_RANGE_EX}', '{_CANON_WEEKLY_LIST_EX}'."
        )

    # Special token: 'rand' = one random day per ISO week
    if any(t == "rand" for t in toks):
        if len(toks) > 1:
            raise ParseError(
                "w:rand cannot be combined with explicit weekdays in the same list. "
                "If you mean 'either random OR Monday', use OR: 'w:rand | w:mon'."
            )
        return  # valid

    for tok in toks:
        if "-" in tok or ":" in tok:
            raise ParseError(
                f"Invalid weekly range '{tok}'. Use '..' (e.g., '{_CANON_WEEKLY_RANGE_EX}')."
            )
        if ".." in tok:
            a, b = tok.split("..", 1)
            if a not in _WEEKDAYS or b not in _WEEKDAYS:
                raise ParseError(
                    f"Unknown weekday in range '{tok}'. "
                    f"Preferred range form is '..' (e.g., '{_CANON_WEEKLY_RANGE_EX}')."
                )
        else:
            if tok not in _WEEKDAYS:
                raise ParseError(
                    f"Unknown weekday token '{tok}'. "
                    f"Use mon..sun (e.g., '{_CANON_WEEKLY_RANGE_EX}' or '{_CANON_WEEKLY_LIST_EX}')."
                )


def _validate_monthly_spec(spec: str):
    """
    Valid monthly tokens (comma-separated):
      - Day of month:           '1'..'31' or negative '-1'..'-31'  (-1 = last day)
      - Day range:              'A..B' where A,B are +/- integers, e.g., '1..7', '-3..-1'
      - Nth weekday:            '2nd-mon', 'last-fri', '-2nd-fri' (hyphen optional; st/nd/rd/th allowed)
      - Business-day ordinal:   'kbd' where k is +/- integer (e.g., '5bd', '-1bd')
    Constraints:
      - 0 is not a valid index for any numeric form
      - |A|,|B|,|k| ≤ 31 (upper bound for validation; actual month length is handled at expansion)
      - nth-weekday number must be 1..5 (or negative -1..-5), or 'last' (≡ -1)
    """
    spec = _expand_monthly_aliases(spec)
    toks = _split_csv_lower(spec)
    if not toks:
        raise ParseError("Empty monthly spec")

    for tok in toks:
        # 1) Day-of-month (positive or negative)
        if _int_like_re.fullmatch(tok):
            try:
                n = int(tok)
            except Exception:
                raise ParseError(f"Invalid day-of-month '{tok}'.")
            if n == 0:
                raise ParseError(
                    "Day-of-month 0 is not allowed. Use 1..31 or negative -1..-31 (e.g., -1 for last day)."
                )
            if abs(n) > 31:
                raise ParseError(
                    f"Day-of-month '{tok}' out of range. Use 1..31 or -1..-31."
                )
            continue

        # 2) Day range A..B (each side may be negative).
        if ".." in tok:
            a_s, b_s = tok.split("..", 1)
            if not (_int_like_re.fullmatch(a_s) and _int_like_re.fullmatch(b_s)):
                raise ParseError(
                    f"Invalid monthly range '{tok}'. Use integer endpoints, e.g., '1..7' or '-3..-1'."
                )
            a, b = int(a_s), int(b_s)
            if a == 0 or b == 0:
                raise ParseError(
                    f"Monthly range '{tok}' uses 0 which is not a valid day index."
                )
            if abs(a) > 31 or abs(b) > 31:
                raise ParseError(
                    f"Monthly range '{tok}' out of bounds. Use values within -31..31."
                )
            continue

        if ":" in tok:
            raise ParseError(
                f"Invalid monthly range '{tok}'. Use '..' (e.g., '1..7' or '-3..-1')."
            )

        # 3) Nth-weekday: '2nd-mon', 'last-fri', '-2nd-tue'
        m = _nth_weekday_re.match(tok)
        if m:
            n_raw, wd = m.group(1), m.group(2)
            if n_raw == "last":
                # ok (equivalent to -1)
                continue
            # strip any ordinal suffix
            n_txt = re.sub(r"(st|nd|rd|th)$", "", n_raw)
            try:
                k = int(n_txt)
            except Exception:
                raise ParseError(
                    f"Invalid nth-weekday number '{n_raw}' in token '{tok}'. "
                    f"Use 1..5 or 'last' (e.g., '2nd-mon', 'last-fri')."
                )
            if k == 0 or abs(k) > 5:
                suggestion = f"last-{wd}" if k > 5 else None
                msg = "nth-weekday must be between 1 and 5 (or 'last')."
                if suggestion:
                    msg += f" Did you mean '{suggestion}'?"
                raise ParseError(f"{msg} Offending token: '{tok}'.")
            continue

        # 4) Business-day ordinal: 'kbd' or '-kbd'
        m = _bd_re.match(tok)
        if m:
            k = int(m.group(1))
            if k == 0:
                raise ParseError(
                    "Business-day index 0 is not allowed. Use 1..31 or -1..-31 (e.g., -1bd for last business day)."
                )
            if abs(k) > 31:
                raise ParseError(
                    f"Business-day index '{k}' out of range. Use values within -31..31."
                )
            continue

        # 5) Unknown
        raise ParseError(
            f"Unknown monthly token '{tok}'. Examples: "
            f"'15', '-1', '1..7', '-3..-1', '2nd-mon', 'last-fri', '5bd'."
        )


def _validate_yearly_token(tok: str):
    """Validate individual yearly token."""
    tok = tok.strip().lower()
    if tok in _QUARTERS or re.fullmatch(r"q[1-4][sme]", tok):
        return
    if ":" in tok:
        raise ParseError(
            f"Invalid yearly range '{tok}'. Use '..' (e.g., 'y:07-01..07-31', 'y:q1..q2')."
        )
    if ".." in tok:
        a, b = tok.split("..", 1)
        pa = _parse_y_token(a)
        pb = _parse_y_token(b)
        if not pa or not pb:
            raise ParseError(f"Invalid yearly range '{tok}'")
        return
    p = _parse_y_token(tok)
    if not p:
        raise ParseError(f"Unknown yearly token '{tok}'")


def _validate_yearly_spec(spec: str):
    """
    Valid yearly tokens (comma-separated):
      - Single day:        'DD-MM'                     (e.g., '25-12')
      - Day range:         'DD-MM..DD-MM'              (inclusive; e.g., '01-03..31-03')
      - Quarter alias:     'q1'..'q4', 'q1s/q1m/q1e'    (quarter window or start/mid/end month)
      - Quarter range:     'qX..qY' (X<=Y)             (e.g., 'q1..q2' → Jan–Jun)
      - Quarter range:     'qXs:qYs' (suffix must match)

    Friendly suggestions are provided for common mistakes (non-padded, month-only, cross-year, etc).
    """

    toks = _split_csv_lower(spec)
    if not toks:
        raise ParseError("Empty yearly spec")

    MONTH_MAX = {
        1: 31,
        2: 29,
        3: 31,
        4: 30,
        5: 31,
        6: 30,
        7: 31,
        8: 31,
        9: 30,
        10: 31,
        11: 30,
        12: 31,
    }
    _quarter_re = re.compile(r"^q[1-4][sme]?$")
    _quarter_range_re = re.compile(r"^(q[1-4])([sme])?\.\.(q[1-4])([sme])?$")

    def _last_day(mm: int) -> int:
        return MONTH_MAX.get(mm, 31)

    def _check_day_month(dd: int, mm: int, label: str, tok: str):
        if mm < 1 or mm > 12:
            raise ParseError(
                f"Invalid month '{mm:02d}' in '{tok}' ({label}). Months must be 01..12."
            )
        maxd = _last_day(mm)
        if dd < 1 or dd > maxd:
            near = maxd if dd > maxd else 1
            hint = f" {_MONTH_FULL[mm]} has {maxd} days."
            sug1 = f"{near:02d}-{mm:02d}"
            sug2 = f"01-{mm:02d}..{maxd:02d}-{mm:02d}"
            raise ParseError(
                f"Invalid day '{dd:02d}' for month '{mm:02d}' in '{tok}' ({label}).{hint} "
                f"Try '{sug1}' or '{sug2}'."
            )

    _month_only = re.compile(r"^\d{1,2}$")  # '3' or '03'
    _month_range_only = re.compile(r"^\d{1,2}\.\.\d{1,2}$")  # '3..4'
    _non_padded_dm = re.compile(r"^\d{1,2}-\d{1,2}(?:\.\.\d{1,2}-\d{1,2})?$")
    _padded_dm = re.compile(
        r"^(?P<d1>\d{2})-(?P<m1>\d{2})(?:\.\.(?P<d2>\d{2})-(?P<m2>\d{2}))?$"
    )

    for tok in toks:
        if ":" in tok:
            raise ParseError(
                "Yearly ranges must use '..' (e.g., '01-01..12-31', 'q1..q2')."
            )
        # --- Quarters (single) ---
        if _quarter_re.fullmatch(tok):
            continue

        # --- Quarter ranges like 'q1..q2' (monotonic only) ---
        m = _quarter_range_re.fullmatch(tok)
        if m:
            q_from = int(m.group(1)[1])
            q_to = int(m.group(3)[1])
            suf_from = m.group(2) or ""
            suf_to = m.group(4) or ""
            if suf_from != suf_to:
                raise ParseError(
                    f"Invalid quarter range '{tok}': suffixes must match "
                    "(use q1s..q2s or q1..q2)."
                )
            if q_to < q_from:
                raise ParseError(
                    f"Invalid quarter range '{tok}': end quarter precedes start quarter. "
                    f"Split across the year boundary, e.g., 'q{q_from}, q{q_to}'."
                )
            continue

        # --- Month-only like '03' → suggest full month ---
        if _month_only.match(tok):
            mm = int(tok)
            if not (1 <= mm <= 12):
                raise ParseError(
                    f"Invalid month '{tok}'. Months must be 01..12. "
                    f"Try '{mm:02d}' or the full-month form '01-{mm:02d}..{_last_day(mm):02d}-{mm:02d}'."
                )
            raise ParseError(
                f"Yearly token '{tok}' is incomplete. Did you mean the full month? "
                f"Try '01-{mm:02d}..{_last_day(mm):02d}-{mm:02d}'."
            )

        # --- 'MM..MM' → suggest full multi-month range with proper end day ---
        if _month_range_only.match(tok):
            m1, m2 = (int(x) for x in tok.split("..", 1))
            if not (1 <= m1 <= 12 and 1 <= m2 <= 12):
                raise ParseError(
                    f"Invalid month range '{tok}'. Months must be 01..12. "
                    f"Try '01-{m1:02d}..{_last_day(m2):02d}-{m2:02d}'."
                )
            if m2 < m1:
                left = f"01-{m1:02d}..31-12"
                right = f"01-01..{_last_day(m2):02d}-{m2:02d}"
                raise ParseError(
                    f"Invalid month range '{tok}': end month is before start month. "
                    f"Split across years, e.g., '{left}, {right}'."
                )
            raise ParseError(
                f"Yearly token '{tok}' is incomplete. Did you mean a full multi-month range? "
                f"Try '01-{m1:02d}..{_last_day(m2):02d}-{m2:02d}'."
            )

        # --- Zero-padding guidance for DM or DM..DM ---
        if _non_padded_dm.match(tok) and not _padded_dm.match(tok):
            pieces = re.split(r"-|\\.\\.", tok)
            padded = "..".join(
                [f"{int(pieces[0]):02d}-{int(pieces[1]):02d}"]
                + (
                    [f"{int(pieces[2]):02d}-{int(pieces[3]):02d}"]
                    if len(pieces) == 4
                    else []
                )
            )
            raise ParseError(
                f"Invalid yearly token '{tok}'. Use zero-padded 'DD-MM' or 'DD-MM..DD-MM'. "
                f"Try '{padded}'."
            )

        # --- Fully padded DM or DM..DM → validate content/order ---
        m = _padded_dm.match(tok)
        if not m:
            raise ParseError(
                f"Unknown yearly token '{tok}'. Expected 'DD-MM', 'DD-MM..DD-MM', "
                f"or quarter aliases 'q1..q4'/'q1s/q1m/q1e' (e.g., 'q1', 'q1s', 'q1..q2')."
            )

        d1 = int(m.group("d1"))
        m1 = int(m.group("m1"))
        d2g = m.group("d2")
        m2g = m.group("m2")

        if d2g is None and m2g is None:
            _check_day_month(d1, m1, "single day", tok)
            continue

        d2 = int(d2g)
        m2 = int(m2g)
        _check_day_month(d1, m1, "range start", tok)
        _check_day_month(d2, m2, "range end", tok)
        if (m2, d2) < (m1, d1):
            left = f"{d1:02d}-{m1:02d}..31-12"
            right = f"01-01..{d2:02d}-{m2:02d}"
            raise ParseError(
                f"Invalid range '{tok}': start must be on/before end; cross-year ranges "
                f"aren't supported. Try splitting: '{left}, {right}'."
            )


def validate_anchor_expr_strict(expr):
    """
    Validate an anchor expression. Accepts:
      - str  (e.g., "w/2:sun + m:1st-mon"), parsed to DNF
      - DNF  (list[list[dict]]), already parsed

    Returns the normalized DNF on success; raises ParseError on failure.
    """
    # 1) Normalize to DNF
    if isinstance(expr, str):
        s = (expr or "").strip()
        if not s:
            raise ParseError("Empty anchor expression.")
        try:
            dnf = parse_anchor_expr_to_dnf_cached(s)
        except ParseError as e:
            raise ParseError(f"{e} (expr: {s})")
    elif isinstance(expr, (list, tuple)):
        dnf = expr
    else:
        raise ParseError(f"Invalid anchor type {type(expr).__name__}; expected string or parsed DNF.")

    # FIX: Handle case where parse_anchor_expr_to_dnf might return (message, [])
    if isinstance(dnf, tuple) and len(dnf) == 2 and isinstance(dnf[0], str):
        # This is an error message tuple from the parser
        raise ParseError(dnf[0])

    # 2) Defensive structure checks (avoid calling .get on strings)
    if not isinstance(dnf, (list, tuple)):
        raise ParseError("Internal error: DNF must be a list of terms.")
    for term in dnf:
        if not isinstance(term, (list, tuple)):
            raise ParseError("Internal error: each term must be a list of atoms.")
        for atom in term:
            if not isinstance(atom, dict):
                raise ParseError("Internal error: each atom must be a dict.")
            if not _is_atom_like(atom):
                raise ParseError("Internal error: atom missing required fields (typ/spec/mods).")

    # 3) Per-atom strict checks
    for term in dnf:
        for a in term:
            typ = (a.get("typ") or a.get("type") or "").lower()
            spec = (a.get("spec") or a.get("value") or "").lower()
            ival = int(a.get("ival") or a.get("intv") or 1)
            mods = a.get("mods") or {}
            active = None

            if typ == "w":
                _validate_weekly_spec(spec)

            elif typ == "m":
                if spec == "rand":
                    active = _active_mod_keys(mods)
                    bad = [k for k in active if k not in ("t", "bd", "wd")]
                    if bad:
                        raise ParseError(f"m:rand does not support @{', '.join(bad)}")
                    ival = int(a.get("ival") or a.get("intv") or 1)
                    if ival < 1:
                        raise ParseError("Monthly interval (/N) must be >= 1")
                else:
                    _validate_monthly_spec(spec)

            elif typ == "y":
                if spec == "rand" or spec.startswith("rand-"):
                    if spec.startswith("rand-"):
                        try:
                            mm = int(spec.split("-", 1)[1])
                        except Exception:
                            raise ParseError(f"Invalid token 'y:{spec}'")
                        if not (1 <= mm <= 12):
                            raise ParseError(f"Invalid month in y:{spec}")
                    if active is None:
                        active = _active_mod_keys(mods)
                    bad = [k for k in active if k not in ("t", "bd", "wd")]
                    if bad:
                        raise ParseError(f"y:{spec} does not support @{', '.join(bad)}")
                else:
                    _validate_yearly_token_format(spec)

            else:
                raise ParseError(f"Unknown anchor type '{typ}'")

    return dnf


# -------- Cached Expansion Functions ----------
@_ttl_lru_cache(maxsize=128)
def expand_weekly_cached(spec: str):
    """Cached expansion of weekly specification to weekday numbers.

    Note: This function is used as a performance primitive; keep it aligned with
    the strict '..' range delimiter contract.
    """
    return sorted(_weekly_spec_to_wset(spec, mods=None))


@_ttl_lru_cache(maxsize=128)
def expand_weekly_cached_mods(spec: str, bd_only: bool):
    """Cached expansion of weekly specification with bd/wd filtering applied."""
    days = expand_weekly_cached(spec)
    if bd_only:
        days = [d for d in days if d < 5]
    return days


@_ttl_lru_cache(maxsize=128)
def expand_yearly_cached(spec: str, y: int):
    """
    Expand yearly tokens into concrete dates for year y.
    Honors ANCHOR_YEAR_FMT == 'DM' or 'MD'.
    - Single dates (e.g., 02-29) are STRICT: if invalid in year y → no date.
    - Ranges (e.g., 01-02..03-31) clamp endpoints to that year's month lengths,
      so whole-month windows stay sensible in non-leap years.
    """
    # Normalize month-name tokens ('mar', 'sep', 'mar..may') to numeric DM/MD ranges
    spec = _rewrite_month_names_to_ranges(spec)
    if not spec:
        return []

    def _mlen(mm: int) -> int:
        return month_len(y, mm)

    def _strict_date(d: int, m: int) -> date | None:
        # For single dates: do NOT clamp; skip invalid combos (e.g., 29 Feb in non-leap years).
        if not (1 <= m <= 12): return None
        if not (1 <= d <= _mlen(m)): return None
        try:
            return date(y, m, d)
        except Exception:
            return None

    def _clamped_date(d: int, m: int) -> date | None:
        # For range endpoints only: clamp inside valid month length.
        if not (1 <= m <= 12): return None
        d = max(1, min(d, _mlen(m)))
        try:
            return date(y, m, d)
        except Exception:
            return None

    def _pair(a: int, b: int) -> tuple[int, int]:
        # Interpret according to ANCHOR_YEAR_FMT; return (day, month)
        return (b, a) if _yearfmt() == "MD" else (a, b)

    days = []
    tokens = _split_csv_lower(spec)

    for tok in tokens:
        m = re.fullmatch(r"(\d{2})-(\d{2})(?:\.\.(\d{2})-(\d{2}))?$", tok)
        if not m:
            # Quarters and month-name windows should be rewritten earlier; ignore others.
            continue

        a, b = int(m.group(1)), int(m.group(2))
        if m.group(3):
            # Range: clamp endpoints
            c, d = int(m.group(3)), int(m.group(4))
            d1, m1 = _pair(a, b)
            d2, m2 = _pair(c, d)
            start = _clamped_date(d1, m1)
            end   = _clamped_date(d2, m2)
            if not start or not end or end < start:
                continue
            cur = start
            while cur <= end:
                days.append(cur)
                cur += timedelta(days=1)
        else:
            # Single date: strict
            d1, m1 = _pair(a, b)
            dd = _strict_date(d1, m1)
            if dd:
                days.append(dd)

    return sorted(days)



@_ttl_lru_cache(maxsize=128)
def expand_monthly_cached(spec: str, y: int, m: int):
    """Cached expansion of monthly specification for given month."""
    out = set()
    last = month_len(y, m)
    spec = _expand_monthly_aliases(spec)

    def resolve_num(n):
        if n < 0:
            k = last + 1 + n
            return k if 1 <= k <= last else None
        return n if 1 <= n <= last else None

    def nth_weekday(n: int, wd: int):
        if n == 0:
            return None
        if n > 0:
            d = date(y, m, 1)
            off = (wd - d.weekday()) % 7
            d = d + timedelta(days=off + (n - 1) * 7)
            return d.day if d.month == m else None
        d = date(y, m, last)
        off = (d.weekday() - wd) % 7
        d = d - timedelta(days=off + (abs(n) - 1) * 7)
        return d.day if d.month == m else None

    def nth_business_day(n: int):
        if n == 0:
            return None
        if n > 0:
            cnt = 0
            d = date(y, m, 1)
            while d.month == m:
                if d.weekday() < 5:
                    cnt += 1
                    if cnt == n:
                        return d.day
                d = d + timedelta(days=1)
            return None
        cnt = 0
        d = date(y, m, last)
        while d.month == m:
            if d.weekday() < 5:
                cnt += 1
                if cnt == abs(n):
                    return d.day
            d = d - timedelta(days=1)
        return None

    for tok in _split_csv_lower(spec):
        m1 = _nth_weekday_re.match(tok)
        if m1:
            n_raw, wd_s = m1.group(1), m1.group(2)
            if n_raw == "last":
                n = -1
            else:
                n_txt = re.sub(r"(st|nd|rd|th)$", "", n_raw)
                n = int(n_txt)
            d0 = nth_weekday(n, _WEEKDAYS[wd_s])
            if d0:
                out.add(d0)
            continue
        m2 = _bd_re.match(tok)
        if m2:
            n = int(m2.group(1))
            d0 = nth_business_day(n)
            if d0:
                out.add(d0)
                continue
        if ".." in tok:
            a_raw, b_raw = tok.split("..", 1)
            a_raw = int(a_raw)
            b_raw = int(b_raw)
            a = resolve_num(a_raw)
            b = resolve_num(b_raw)
            if a is None or b is None:
                continue
            step = 1 if a <= b else -1
            for r in range(a, b + step, step):
                out.add(r)
        else:
            try:
                n = int(tok)
                r = resolve_num(n)
                if r:
                    out.add(r)
            except:
                pass
    return sorted(out)


def expand_monthly_for_month(spec: str, y: int, m: int):
    """Wrapper for cached monthly expansion."""
    return expand_monthly_cached(spec, y, m)


def expand_weekly(spec: str):
    """Wrapper for cached weekly expansion."""
    return expand_weekly_cached(spec)


def expand_yearly_for_year_strict(spec: str, y: int):
    """Wrapper for cached yearly expansion."""
    return expand_yearly_cached(spec, y)


# -------- Rolls / atoms ----------
def roll_apply(dt: date, mods: dict) -> date:
    roll = mods.get("roll")  # 'pbd' | 'nbd' | 'nw' | 'next-wd' | 'prev-wd' | None

    if roll in ("pbd", "nbd", "nw"):
        if dt.weekday() > 4:  # weekend
            if roll == "pbd":
                for _ in range(8):
                    if dt.weekday() <= 4:
                        break
                    dt -= timedelta(days=1)
                else:
                    raise ParseError("roll_apply: failed to reach business day (pbd)")
            elif roll == "nbd":
                for _ in range(8):
                    if dt.weekday() <= 4:
                        break
                    dt += timedelta(days=1)
                else:
                    raise ParseError("roll_apply: failed to reach business day (nbd)")
            else:  # 'nw'
                prev_dt = dt
                next_dt = dt
                for _ in range(8):
                    if prev_dt.weekday() <= 4:
                        break
                    prev_dt -= timedelta(days=1)
                else:
                    raise ParseError("roll_apply: failed to reach business day (nw prev)")
                for _ in range(8):
                    if next_dt.weekday() <= 4:
                        break
                    next_dt += timedelta(days=1)
                else:
                    raise ParseError("roll_apply: failed to reach business day (nw next)")
                if (dt - prev_dt) <= (next_dt - dt):
                    dt = prev_dt
                else:
                    dt = next_dt

    elif roll in ("next-wd", "prev-wd"):
        tgt = mods.get("wd")
        if tgt is not None:
            if roll == "next-wd":
                for _ in range(8):
                    if dt.weekday() == tgt:
                        break
                    dt += timedelta(days=1)
                else:
                    raise ParseError("roll_apply: failed to reach target weekday (next-wd)")
            else:  # 'prev-wd'
                for _ in range(8):
                    if dt.weekday() == tgt:
                        break
                    dt -= timedelta(days=1)
                else:
                    raise ParseError("roll_apply: failed to reach target weekday (prev-wd)")

    return dt

def _weeks_between(d1: date, d2: date) -> int:
    """Return number of ISO weeks between two dates (d2 - d1)."""
    # Convert both dates to Monday of their ISO week
    iso1 = d1.isocalendar()
    iso2 = d2.isocalendar()
    mon1 = date.fromisocalendar(iso1.year, iso1.week, 1)
    mon2 = date.fromisocalendar(iso2.year, iso2.week, 1)
    # Compute difference in weeks
    return (mon2 - mon1).days // 7

def apply_day_offset(d: date, mods: dict) -> date:
    """Apply day offset modifier."""
    off = int(mods.get("day_offset", 0) or 0)
    return d + timedelta(days=off) if off else d


def base_next_after_atom(atom, ref_d: date) -> date:
    """Find next date after ref_d that matches atom (without mods)."""
    typ = (atom.get("typ") or "").lower()
    spec = (atom.get("spec") or "").lower()
    mods = atom.get("mods") or {}

    # Handle weekly random first
    if typ == "w" and "rand" in spec:
        # Pick exactly one deterministic day per ISO week, strictly after ref_d
        p = ref_d + timedelta(days=1)
        for _ in range(366):  # safety
            iso = p.isocalendar()
            dow = _weekly_rand_pick(iso.year, iso.week, mods)
            mon = _week_monday(p)
            dt = mon + timedelta(days=dow)
            if dt > ref_d:
                return dt
            # move to next ISO week
            p = mon + timedelta(days=7, seconds=1)
        return ref_d + timedelta(days=7)  # fallback

    # Normal weekly path (non-rand)
    if typ == "w":
        bd_only = bool(mods.get("bd") or (mods.get("wd") is True))
        days = expand_weekly_cached_mods(spec, bd_only)
        if not days:  # No weekdays in spec
            # Fallback: return far future to avoid infinite loop
            return ref_d + timedelta(days=365)
        
        for i in range(1, 15):  # next 2 weeks is plenty
            cand = ref_d + timedelta(days=i)
            if cand.weekday() in days:
                return cand
        return ref_d + timedelta(days=7)

    if typ == "m":
        y, m = ref_d.year, ref_d.month
        # Union across comma-separated monthly tokens (each token → a set of DOMs)
        tokens = _split_csv_tokens(spec)
        for _ in range(24):  # scan up to 24 months out
            doms_union = set()
            for tok in tokens:
                try:
                    for d0 in expand_monthly_cached(tok, y, m):
                        doms_union.add(d0)
                except Exception:
                    # ignore a single bad token here; validation should have blocked already
                    pass
            for d0 in sorted(doms_union):
                cand = date(y, m, d0)
                if cand > ref_d:
                    return cand
            # advance month
            m = 1 if m == 12 else (m + 1)
            if m == 1:
                y += 1
        # safe fallback
        return ref_d + timedelta(days=365)

    if typ == "y":
        y = ref_d.year
        # Look far enough ahead to catch leap windows etc.
        for _ in range(12):
            days = expand_yearly_cached(spec, y)
            for cand in days:
                if cand > ref_d:
                    return cand
            y += 1
        # safe fallback (avoid .replace on 29-Feb)
        return ref_d + timedelta(days=366)


# ------------------------------------------------------------------------------
# Anchor scheduling & iteration helpers
# ------------------------------------------------------------------------------
def next_after_atom_with_mods(atom, ref_d: date, default_seed: date) -> date:
    """
    Strictly-after guard + /N gating by buckets:
      - Weekly: ISO week buckets
      - Monthly: *valid-month* buckets (months that actually have a match for 'spec')
      - Yearly: calendar year buckets
    Then applies roll + day offsets.
    """
    ival = int(atom.get("ival", 1) or 1)
    if ival > 100:
        ival = 100

    seed = default_seed or ref_d
    probe = ref_d
    typ   = atom["typ"]
    mods  = atom.get("mods") or {}
    spec  = atom.get("spec") or ""

    # Fast path: ival==1 and no modifiers
    if ival == 1 and not _active_mod_keys(mods):
        candidate = base_next_after_atom(atom, ref_d)
        if candidate > ref_d:
            return candidate

    # ---- helpers for weekly/yearly gating ----
    def _allowed_by_interval(cand: date) -> bool:
        if ival <= 1:
            return True
        if typ == "w":
            # Use proper week difference
            weeks_diff = _weeks_between(seed, cand)
            return weeks_diff % ival == 0
        if typ == "y":
            return (_year_index(cand) - _year_index(seed)) % ival == 0
        # monthly handled separately
        return True

    def _advance_probe_to_next_allowed_bucket(cand: date) -> date:
        if ival <= 1:
            return cand
        if typ == "w":
            cur_monday = cand - timedelta(days=cand.weekday())
            weeks_from_seed = _weeks_between(seed, cur_monday)
            diff = weeks_from_seed % ival
            add_weeks = (ival - diff) if diff != 0 else 0
            next_allowed_monday = cur_monday + timedelta(weeks=add_weeks or ival)
            return next_allowed_monday - timedelta(days=1)
        if typ == "y":
            diff = (_year_index(cand) - _year_index(seed)) % ival
            add_y = (ival - diff) if diff != 0 else 0
            next_jan1 = date(cand.year + (add_y or ival), 1, 1)
            return next_jan1 - timedelta(days=1)
        return cand

    # ---- helpers for monthly valid-month gating ----
    def _month_doms(y: int, m: int) -> list[int]:
        try:
            return sorted(expand_monthly_cached(spec, y, m))
        except Exception:
            return []

    def _month_has_hit(y: int, m: int) -> bool:
        return bool(_month_doms(y, m))

    def _first_hit_after_probe_in_month(y: int, m: int, p: date) -> date | None:
        for d0 in _month_doms(y, m):
            dt = date(y, m, d0)
            if dt > p:
                return dt
        return None

    def _next_valid_month_on_or_after(y: int, m: int) -> tuple[int, int]:
        yy, mm = y, m
        for _ in range(480):  # safety bound
            if _month_has_hit(yy, mm):
                return yy, mm
            mm += 1
            if mm > 12:
                yy += 1
                mm = 1
        return y, m

    def _advance_k_valid_months(start_y: int, start_m: int, k: int) -> tuple[int, int]:
        yy, mm = start_y, start_m
        steps = max(k, 0)
        while steps >= 0:
            mm += 1
            if mm > 12:
                mm = 1
                yy += 1
            yy, mm = _next_valid_month_on_or_after(yy, mm)
            steps -= 1
        return yy, mm

    # ---- guarded iteration ----
    for _ in range(MAX_ANCHOR_ITER):
        base = base_next_after_atom(atom, probe)

        # @bd modifier: weekdays only - skip weekends entirely (move to next bucket)
        if mods.get("bd") and base.weekday() > 4:  # 5=Saturday, 6=Sunday
            probe = base + timedelta(days=1)
            continue

        # weekly/yearly gating first
        if typ in ("w", "y") and not _allowed_by_interval(base):
            probe = _advance_probe_to_next_allowed_bucket(base)
            continue

        # monthly special-case: /N by *valid months*
        if typ == "m" and ival > 1:
            by, bm = base.year, base.month

            # seed bucket = first valid month on/after seed
            sy, sm = _next_valid_month_on_or_after(seed.year, seed.month)

            # ensure base is in a valid month and strictly > probe
            if not _month_has_hit(by, bm):
                by, bm = _next_valid_month_on_or_after(by, bm)
                nxt = _first_hit_after_probe_in_month(by, bm, probe)
                if nxt is None:
                    ny, nm = _advance_k_valid_months(by, bm, 0)
                    doms = _month_doms(ny, nm)
                    base = date(ny, nm, doms[0])
                else:
                    base = nxt
            else:
                if base <= probe:
                    nxt = _first_hit_after_probe_in_month(by, bm, probe)
                    if nxt is None:
                        ny, nm = _advance_k_valid_months(by, bm, 0)
                        doms = _month_doms(ny, nm)
                        base = date(ny, nm, doms[0])
                    else:
                        base = nxt

            # count valid-month steps from (sy,sm) to base(y,m)
            cnt = 0
            ty, tm = sy, sm
            while (ty, tm) != (base.year, base.month) and cnt < 480:
                ty, tm = _advance_k_valid_months(ty, tm, 0)
                cnt += 1

            if (cnt % ival) != 0:
                steps = ival - (cnt % ival)
                ny, nm = _advance_k_valid_months(base.year, base.month, steps - 1)
                doms = _month_doms(ny, nm)
                base = date(ny, nm, doms[0])

        # --- apply roll + offsets and decide acceptance ---
        rolled = roll_apply(base, mods)
        cand = apply_day_offset(rolled, mods)

        roll_kind = mods.get("roll")
        if roll_kind in ("pbd", "nbd", "nw"):
            # Business-day rolls:
            #   base must still be strictly after ref_d,
            #   but the rolled candidate is allowed to land ON ref_d.
            if base > ref_d and cand >= ref_d:
                return cand
        else:
            # Everything else keeps strict "after ref_d" semantics.
            if cand > ref_d:
                return cand

        # advance probe for next iteration
        probe = base + timedelta(days=1)

    # fallback
    if os.environ.get("NAUTICAL_DIAG") == "1":
        _warn_once_per_day(
            "next_after_atom_fallback",
            f"[nautical] next_after_atom_with_mods fallback after {MAX_ANCHOR_ITER} iterations.",
        )
    return ref_d + timedelta(days=365)




def atom_matches_on(atom, d: date, default_seed: date) -> bool:
    """Check if atom matches on specific date (with roll window)."""
    # Reduced window from 7 to 5 days
    for k in range(1, 6):
        if next_after_atom_with_mods(atom, d - timedelta(days=k), default_seed) == d:
            return True
    return False


def next_after_term(term, ref_d: date, default_seed: date):
    """Find next date after ref_d that matches all atoms in term.

    Fast path for single-atom terms to avoid unnecessary intersection logic
    and /N-gating artifacts.
    """
    # ---------- Fast path: single atom ----------
    if len(term) == 1:
        atom = term[0]
        nxt = next_after_atom_with_mods(atom, ref_d, default_seed)
        mods = atom.get("mods") or {}
        hhmm = mods.get("t")
        return nxt, hhmm

    # ---------- General case: intersection of multiple atoms ----------
    cur = ref_d
    for _ in range(min(INTERSECTION_GUARD_STEPS, 100)):
        # For each atom, find its next candidate >= cur
        cands = [next_after_atom_with_mods(a, cur, default_seed) for a in term]
        nxt = max(cands)

        # Check that all atoms "match" on this day (with roll window, etc.)
        if all(atom_matches_on(a, nxt, default_seed) for a in term):
            hhmm = None
            for a in term:
                mods = a.get("mods") or {}
                if mods.get("t"):
                    tval = mods["t"]
                    if isinstance(tval, list):
                        hhmm = tval[0] if tval else None
                    else:
                        hhmm = tval
                    break
            return nxt, hhmm

        # Otherwise advance the search window
        cur = nxt

    # Guard-rail fallback to avoid infinite loops on pathological patterns
    return ref_d + timedelta(days=365), None



def _is_simple_weekly(dnf):
    """Fast path check for simple weekly expressions."""
    if len(dnf) != 1 or len(dnf[0]) != 1:
        return False
    atom = dnf[0][0]
    return (
        atom["typ"] == "w"
        and "rand" not in (atom.get("spec") or "")
        and atom.get("ival", 1) == 1
        and not _active_mod_keys(atom.get("mods"))
    )


def _simple_weekly_next(after_date: date, weekdays: list) -> date:
    """Optimized next date calculation for simple weekly patterns."""
    current_wd = after_date.weekday()
    for offset in range(1, 8):  # Check next 7 days
        cand = after_date + timedelta(days=offset)
        if cand.weekday() in weekdays:
            return cand
    # Fallback: should not happen with proper weekdays list
    return after_date + timedelta(days=7)


def next_after_expr(dnf, after_date, default_seed=None, seed_base=None):
    """
    Return the next matching *local date* strictly > after_date.
    Uses optimized paths for common cases.
    """

    # Fast path for simple weekly expressions
    if _is_simple_weekly(dnf):
        atom = dnf[0][0]
        days = expand_weekly_cached(atom["spec"])
        return _simple_weekly_next(after_date, days), {"basis": "simple_weekly"}

    best = None
    best_meta = None

    for term_id, term in enumerate(dnf):
        rk, info = _term_rand_info(term)

        if rk == "m":
            if any(_atype(a) == "y" for a in term):
                cand = _next_for_and(term, after_date, default_seed)
                if cand and (best is None or cand < best):
                    best, best_meta = cand, {"basis": "rand+yearly"}
                continue

            if seed_base is None:
                seed_base = "preview"
            mods = info.get("mods") or {}
            bd_only = bool(mods.get("bd"))  # <-- only business day flag
            ival = int(info.get("ival") or 1)

            seed_loc = default_seed or after_date
            y, m = after_date.year, after_date.month

            for _ in range(24):  # up to 24 months ahead
                if ival > 1 and ((_months_since(seed_loc, y, m) % ival) != 0):
                    m = 1 if m == 12 else m + 1
                    if m == 1:
                        y += 1
                    continue

                cands = _term_candidates_in_month(term, y, m, info["atom_idx"], bd_only)
                if cands:
                    period_key = f"{y:04d}{m:02d}"
                    seed_key = f"{seed_base}|m|{term_id}|{period_key}"
                    idx = _sha_pick(len(cands), seed_key)
                    choice = cands[idx]
                    if choice > after_date:
                        cand, meta = choice, {
                            "basis": "rand",
                            "rand_period": period_key,
                        }
                        if best is None or cand < best:
                            best, best_meta = cand, meta
                        break
                m = 1 if m == 12 else m + 1
                if m == 1:
                    y += 1
            continue

        if rk == "y":
            if seed_base is None:
                seed_base = "preview"
            mods = info.get("mods") or {}
            bd_only = bool(mods.get("bd"))
            target_m = info.get("month", None)
            y = after_date.year

            for _ in range(10):  # up to 10 years ahead
                if target_m is None:
                    # NEW: plain y:rand → gather all monthly candidates for this year
                    cands = []
                    for mm in range(1, 13):
                        cands.extend(_term_candidates_in_month(term, y, mm, info["atom_idx"], bd_only))
                    period_key = f"{y:04d}"
                else:
                    cands = _term_candidates_in_month(term, y, int(target_m), info["atom_idx"], bd_only)
                    period_key = f"{y:04d}-{int(target_m):02d}"

                if cands:
                    seed_key = f"{seed_base}|y|{term_id}|{period_key}"
                    idx = _sha_pick(len(cands), seed_key)
                    choice = cands[idx]
                    if choice > after_date:
                        cand, meta = choice, {"basis": "rand", "rand_period": period_key}
                        if best is None or cand < best:
                            best, best_meta = cand, meta
                        break
                y += 1
            continue

        # normal term
        cand, _ = next_after_term(term, after_date, default_seed)
        if cand and (best is None or cand < best):
            best, best_meta = cand, {"basis": "term"}

    return best, best_meta


def anchors_between_expr(dnf, start_excl, end_excl, default_seed, seed_base=None):
    """Find all matching dates between start_excl and end_excl."""
    # Fast path for empty or very large ranges
    if start_excl >= end_excl:
        return []

    # Use more efficient approach for large date ranges
    if (end_excl - start_excl).days > 365 * 2:
        return _anchors_between_large_range(
            dnf, start_excl, end_excl, default_seed, seed_base
        )

    acc = []
    cur = start_excl
    # More conservative iteration limit
    while len(acc) < min(UNTIL_COUNT_CAP, 500):
        d, _ = next_after_expr(dnf, cur, default_seed, seed_base=seed_base)
        if d is None or d >= end_excl:
            break
        if acc and d <= acc[-1]:
            cur = acc[-1] + timedelta(days=1)
            continue
        acc.append(d)
        cur = d
    return acc


def _anchors_between_large_range(
    dnf, start_excl, end_excl, default_seed, seed_base=None
):
    """Optimized version for large date ranges."""
    acc = []
    cur = start_excl
    batch_size = min(100, UNTIL_COUNT_CAP)

    while len(acc) < UNTIL_COUNT_CAP and cur < end_excl:
        d, _ = next_after_expr(dnf, cur, default_seed, seed_base=seed_base)
        if d is None or d >= end_excl:
            break
        acc.append(d)
        cur = d + timedelta(days=1)  # Small optimization

        # Safety check for too many iterations
        if len(acc) >= batch_size:
            break

    return acc


def expr_has_m_or_y(dnf) -> bool:
    """Check if expression contains monthly or yearly atoms."""
    for term in dnf:
        for a in term:
            if a["typ"] in ("m", "y"):
                return True
    return False


def pick_hhmm_from_dnf_for_date(dnf, target: date, default_seed: date):
    """Extract HH:MM time from DNF for given date.

    If @t contains multiple times, returns the earliest-listed time.
    """
    for term in dnf:
        if all(atom_matches_on(a, target, default_seed) for a in term):
            for a in term:
                tval = a["mods"].get("t")
                if not tval:
                    continue
                if isinstance(tval, list):
                    return tval[0] if tval else None
                return tval
    return None


# ------------------------------------------------------------------------------
# Datetime construction (local wall-clock -> UTC)
# ------------------------------------------------------------------------------
def build_local_datetime(d: date, hhmm=(DEFAULT_DUE_HOUR, 0)) -> datetime:
    """Build a UTC datetime from local wall-clock date+time with DST handling."""
    hh, mm = hhmm
    naive = datetime(d.year, d.month, d.day, hh, mm, 0)
    if not _LOCAL_TZ:
        return naive.replace(tzinfo=timezone.utc)

    candidates = []
    for fold in (0, 1):
        aware = naive.replace(tzinfo=_LOCAL_TZ, fold=fold)
        back = aware.astimezone(timezone.utc).astimezone(_LOCAL_TZ)
        if back.replace(tzinfo=None) == naive:
            candidates.append(aware)
    if candidates:
        # Ambiguous time: choose the earlier UTC instant for determinism.
        best = min(candidates, key=lambda dt: dt.astimezone(timezone.utc))
        return best.astimezone(timezone.utc)

    # Non-existent time (spring forward): shift forward by 1 hour, then to next valid minute.
    cand = naive + timedelta(hours=1)
    for _ in range(180):
        for fold in (0, 1):
            aware = cand.replace(tzinfo=_LOCAL_TZ, fold=fold)
            back = aware.astimezone(timezone.utc).astimezone(_LOCAL_TZ)
            if back.replace(tzinfo=None) == cand:
                return aware.astimezone(timezone.utc)
        cand += timedelta(minutes=1)
    return naive.replace(tzinfo=_LOCAL_TZ, fold=0).astimezone(timezone.utc)



# ------------------------------------------------------------------------------
# Yearly token helpers
# ------------------------------------------------------------------------------
def _iter_y_segments(s: str):
    """
    Yield the raw yearly-spec segments that follow 'y:' up to the next
    term delimiter (+, |, ) or end. We don't fully parse here; it's
    just for linting.
    """
    for m in re.finditer(r'y\s*:\s*([^\+\|\)]*)', s):
        yield (m.group(1) or "").strip()

def lint_anchor_expr(expr: str) -> tuple[str | None, list[str]]:
    # Accept anchors wrapped in single or double quotes and normalize rand-month alias.
    s = _unwrap_quotes(expr or "").strip().lower()
    # normalize 'mm-rand' → 'rand-mm' for consistency
    s = re.sub(r'\b(\d{2})-rand\b', r'rand-\1', s)
    # Allow bare month aliases: replace 'y:jun' with a canonical monthly window for linting
    def _lint_month_alias_sub(m):
        mm = _month_from_alias(m.group(1))
        if not mm: return m.group(0)
        return f"y:{_year_full_month_range_token(mm)}"
    # Allow bare month aliases ONLY when they are not part of a numeric day-month like 'y:01-13'.
    #  - 'y:jan' or 'y:03' → expand to full month window
    #  - do NOT touch 'y:01-13' / 'y:jun-01' etc.
    s = re.sub(r'\by:([a-z]{3})(?=\b(?!-)|[,+|()])', _lint_month_alias_sub, s)
    s = re.sub(r'\by:(\d{2})(?=(?:\b(?!-)|[,+|()]))', _lint_month_alias_sub, s)
    if not s:
        return None, []
    warnings: list[str] = []
    

    # ------------------------------------------------------------------
    # V2 delimiter contract (strict)
    # ------------------------------------------------------------------
    if re.search(r"\bw(?:/\d+)?\s*:\s*(?:mon|tue|wed|thu|fri|sat|sun)\s*-\s*(?:mon|tue|wed|thu|fri|sat|sun)\b", s):
        return ("Weekly ranges must use '..' (e.g., 'w:mon..fri').", [])
    if re.search(r"\bw(?:/\d+)?\s*:\s*(?:mon|tue|wed|thu|fri|sat|sun)(?:\s*:\s*(?:mon|tue|wed|thu|fri|sat|sun))+\b", s):
        return ("Weekly ranges must use '..' (e.g., 'w:mon..fri').", [])


    # 1) Yearly tokens: check only inside y: segments
    fmt = _yearfmt()  # "MD" or "DM"
    for seg in _iter_y_segments(s):
        # split on commas (multiple y tokens)
        for tok in _split_csv_tokens(seg):
            # bare dd:mm (no hyphens anywhere) → definitely wrong (and not a range)
            if re.fullmatch(r'\d{2}:\d{2}', tok):
                return ("Yearly day/month must use '-', not ':'. Try '05-15' (not '05:15').", [])
            if ":" in tok:
                return ("Yearly ranges must use '..' (e.g., '01-01..12-31', 'q1..q2').", [])
            # valid single day?
            if re.fullmatch(r'\d{2}-\d{2}', tok):
                a, b = tok.split("-")
                x, y = int(a), int(b)
                if fmt == "MD":
                    if x > 12 and 1 <= y <= 12:
                        return (f"'{tok}' looks like DD-MM but config expects MM-DD. Try '{y:02d}-{x:02d}'.", [])
                else:  # DM
                    if y > 12 and 1 <= x <= 12:
                        return (f"'{tok}' looks like MM-DD but config expects DD-MM. Try '{y:02d}-{x:02d}'.", [])
                continue
            # valid range? (V2 '..')
            if re.fullmatch(r'\d{2}-\d{2}\.\.\d{2}-\d{2}', tok):
                left, right = tok.split("..", 1)
                a, b = left.split("-", 1)
                c, d = right.split("-", 1)
                x, y, u, v = int(a), int(b), int(c), int(d)
                if fmt == "MD":
                    if x > 12 and 1 <= y <= 12:
                        return (f"'{tok}' starts like DD-MM but config expects MM-DD. "
                                f"Try '{y:02d}-{x:02d}..{v:02d}-{u:02d}'.", [])
                else:
                    if y > 12 and 1 <= x <= 12:
                        return (f"'{tok}' starts like MM-DD but config expects DD-MM. "
                                f"Try '{y:02d}-{x:02d}..{v:02d}-{u:02d}'.", [])
                continue

    # 2) MD/DM confusion (use configured ANCHOR_YEAR_FMT)
    for m in re.finditer(r'\b(\d{2})-(\d{2})(?=([^\d:]|$))', s):
        a, b = int(m.group(1)), int(m.group(2))
        fmt = _yearfmt()  # "MD" or "DM"
        if fmt == "MD":
            if a > 12 and 1 <= b <= 12:
                return (f"'{m.group(0)}' looks like DD-MM but config expects MM-DD. Try '{b:02d}-{a:02d}'.", [])
        else:  # DM
            if b > 12 and 1 <= a <= 12:
                return (f"'{m.group(0)}' looks like MM-DD but config expects DD-MM. Try '{b:02d}-{a:02d}'.", [])

    # 3) invalid weekday names in typical contexts
    _wd = set(_WD_ABBR)  # ["mon","tue","wed","thu","fri","sat","sun"]
    for wd in re.findall(r'\b[a-z]{3,}\b', s):
        if wd in _wd or wd in ("rand", "rand*"):
            continue
        if re.search(rf'(?:^|[\s\+\|,:@-])(w:|@prev-|@next-|last-|1st|2nd|3rd|4th|5th-){wd}\b', s):
            sug = difflib.get_close_matches(wd, list(_wd), n=1, cutoff=0.6)
            if sug:
                return (f"Unknown weekday '{wd}'. Did you mean '{sug[0]}'?", [])

    # 4) nth weekday suffix errors and range
    ord_ok = {"1": "1st", "2": "2nd", "3": "3rd", "4": "4th", "5": "5th"}
    for m in re.finditer(r'\b(\d+)(st|nd|rd|th)-([a-z]+)\b', s):
        n, suff, wd = m.group(1), m.group(2), m.group(3)
        if n not in ord_ok:
            return (f"Invalid ordinal '{n}{suff}'. Only 1st..5th are supported.", [])
        expect = ord_ok[n]
        if f"{n}{suff}" != expect:
            return (f"Did you mean '{expect}-{wd}' instead of '{n}{suff}-{wd}'?", [])

    # 5) unsatisfiable '+' for pure weekly AND like "w:sat + w:mon"
    and_terms = [t.strip() for t in re.split(r'\|', s)]
    for t in and_terms:
        atoms = [a.strip() for a in re.split(r'\+', t)]
        wsets, only_weekly = [], True
        for a in atoms:
            m = re.match(r'^w(?:(/\d+)?):([a-z0-9\-\:\,]+)$', a)
            if not m:
                only_weekly = False
                break
            spec = m.group(2)
            ws = set()
            simple = True
            for tok in _split_csv_tokens(spec):
                if "-" in tok or ":" in tok:
                    simple = False
                    break
                if tok in _wd:
                    ws.add(tok)
            if not simple:
                only_weekly = False
                break
            if ws:
                wsets.append(ws)
        if only_weekly and wsets and not set.intersection(*wsets):
            return ("These anchors joined with '+' don't share any possible date. "
                    "If you meant 'either/or', join them with ',' or '|'.", [])

    # 6) quarter ranges like "q4..q2" backwards
    g = re.search(r'\bq([1-4])\s*\.\.\s*q([1-4])\b', s)
    if g and int(g.group(2)) < int(g.group(1)):
        return ("Invalid quarter range 'qX..qY': end quarter precedes start quarter. "
                "Split across the year boundary, e.g., 'q4, q1'.", [])

    # 7) gentle tip for legacy multi-@t in one atom
    if re.search(r'y:[^|+)]*@t=\d{2}:\d{2},', s):
        warnings.append("Multiple @t times inside a single 'y:' atom; ensure each spec has its own @t or use '|'.")

    return None, warnings



def _rewrite_weekly_multi_time_atoms(s: str) -> str:
    """
    Rewrite patterns like:
        w:mon@t=09:00,fri@t=15:00
    into:
        w:mon@t=09:00 | w:fri@t=15:00

    Rules:
      - Only triggers inside a single weekly atom (starts with 'w:').
      - Splits on top-level commas, but keeps each token's @t with it.
      - Leaves existing '|' and '+' structure intact.
    """
    out = []
    i, n = 0, len(s)

    def flush_atom(prefix: str, body: str):
        # body like "mon@t=09:00,fri@t=15:00"
        parts = _split_csv_tokens(body)
        if len(parts) <= 1:
            out.append(prefix + body)
            return
        # if every part looks like <dow>(@t=..)? then expand with OR
        pat = re.compile(r"^(mon|tue|wed|thu|fri|sat|sun)(@t=\d{2}:\d{2})?$", re.I)
        if all(pat.match(p) for p in parts):
            expanded = " | ".join(f"{prefix}{p}" for p in parts)
            out.append(expanded)
        else:
            out.append(prefix + body)

    while i < n:
        if s[i] == "w":
            # Find the ':' that terminates the weekly head (supports 'w:' and 'w/2:')
            j = i
            depth = 0
            # First, find the colon that ends the head
            colon = -1
            k = i + 1
            while k < n:
                if s[k] == ":":
                    colon = k
                    break
                if s[k] in "|+)(":
                    break
                k += 1
            if colon == -1:
                out.append(s[i])
                i += 1
                continue
            # Now find the end of this atom at top level
            j = colon + 1
            depth = 0
            while j < n:
                c = s[j]
                if c == "(":
                    depth += 1
                elif c == ")":
                    if depth == 0:
                        break
                    depth -= 1
                elif depth == 0 and c in "|+":
                    break
                j += 1
            prefix = s[i:colon+1]    # e.g., "w:" or "w/2:"
            body   = s[colon+1:j]    # after the colon up to op/end
            flush_atom(prefix, body)
            i = j
        else:
            out.append(s[i])
            i += 1
    return "".join(out)

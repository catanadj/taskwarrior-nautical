#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chained next-link spawner for Taskwarrior.

- Works for classic cp (cp/chainMax/chainUntil) and anchors (anchor/anchor_mode).
- Cap logic unified (chainMax, chainUntil -> numeric cap_no).
- Queues child spawn intent; on-exit hook performs `task import -`.
- Timeline is capped and marks (last link).
"""

import sys, json, os, uuid, subprocess, importlib, random, tempfile
import importlib.util
import atexit
import time as _ptime
import sqlite3
from collections import OrderedDict
from pathlib import Path
from datetime import datetime, timedelta, timezone, time
from functools import lru_cache
import re
import time as _time
import threading
from typing import Any, Optional
import stat

_MAX_JSON_BYTES = 10 * 1024 * 1024
NAUTICAL_HOOK_VERSION = "updateE-20260126"


# Optional: DST-aware local TZ helpers (used by some carry-forward variants)
try:
    import zoneinfo as _zoneinfo
except Exception:  # pragma: no cover
    _zoneinfo = None
ZoneInfo = _zoneinfo.ZoneInfo if _zoneinfo is not None else None


# set config show_analytics=false to disable analytics panel entry.

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

_MAX_CHAIN_WALK = 500  # cap for chain summaries/analytics
_MAX_UUID_LOOKUPS = 50  # max individual UUID exports before giving up
_MAX_ITERATIONS = 2000  # prevent infinite loops in stepping functions
_MIN_FUTURE_WARN = 365 * 2  # warn if chain extends >2 years


_MAX_SPAWN_ATTEMPTS = 3
_SPAWN_RETRY_DELAY = 0.1  # seconds between retries
_STABLE_CHILD_UUID_NAMESPACE = uuid.UUID("1f4b2396-df58-5a32-a879-33f0d3fe711f")
# Spawn intent queue guards (override via env for heavy workloads).
# spawn_queue_max_bytes: warn when queue exceeds this size (on-exit drains).
_DEFAULT_SPAWN_QUEUE_MAX_BYTES = 524288
# spawn_queue_drain_max_items remains in config for legacy docs, but is unused here.
_SPAWN_QUEUE_DRAIN_MAX_ITEMS = 200
_DEFAULT_CHAIN_EXPORT_TIMEOUT_BASE = 1.5
_DEFAULT_CHAIN_EXPORT_TIMEOUT_PER_100 = 1.0
_DEFAULT_CHAIN_EXPORT_TIMEOUT_MAX = 12.0

def _env_float(name: str, default: float) -> float:
    try:
        return float(str(os.environ.get(name, "")).strip() or default)
    except Exception:
        return float(default)

# Panel chain index for fast timeline lookups (set per hook run)
_PANEL_CHAIN_BY_LINK = None
_PANEL_CHAIN_BY_SHORT = None
_CHAIN_CACHE_CHAIN_ID = ""
_CHAIN_CACHE: list[dict[str, Any]] = []
_CHAIN_BY_SHORT: dict[str, dict[str, Any]] = {}
_CHAIN_BY_UUID: dict[str, dict[str, Any]] = {}
_CHAIN_CACHE_LOCK = threading.RLock()
_DIAG_STATS = {
    "run_task_calls": 0,
    "run_task_failures": 0,
    "export_uuid_cache_hits": 0,
    "export_uuid_cache_misses": 0,
    "export_full_cache_hits": 0,
    "export_full_cache_misses": 0,
    "tw_get_cache_hits": 0,
    "unexpected_cache_misses": 0,
    "chain_cache_seeded": 0,
    "run_task_seconds": 0.0,
}

_DIAG_START_TS = _ptime.perf_counter()
# ------------------------------------------------------------------------------
# Debug: wait/scheduled carry-forward
# Set debug_wait_sched=true to include carry computations in the feedback panel.
# ------------------------------------------------------------------------------
_DEFAULT_DEBUG_WAIT_SCHED = False
_LAST_WAIT_SCHED_DEBUG: OrderedDict[str, dict[str, Any]] = OrderedDict()
_MAX_WAIT_SCHED_DEBUG = 32
_WARNED_SPAWN_QUEUE_GROWTH = False
_WARNED_CHAIN_EXPORT: set[str] = set()


def _diag(msg: str) -> None:
    try:
        _load_core()
    except Exception:
        pass
    if core is not None:
        core.diag(msg, "on-modify", str(TW_DATA_DIR))
    elif os.environ.get("NAUTICAL_DIAG") == "1":
        try:
            sys.stderr.write(f"[nautical] {msg}\n")
        except Exception:
            pass


def _is_lock_error(stderr: str) -> bool:
    try:
        _load_core()
    except Exception:
        pass
    if core is not None:
        return core.is_lock_error(stderr)
    s = (stderr or "").lower()
    return (
        "database is locked" in s or "unable to lock" in s
        or "resource temporarily unavailable" in s or "another task is running" in s
        or "lock file" in s or "lockfile" in s or "locked by" in s or "timeout" in s
    )


def _tw_lock_path() -> Path:
    return TW_DATA_DIR / "lock"


def _tw_lock_recent(max_age_s: float = 5.0) -> bool:
    try:
        p = _tw_lock_path()
        if not p.exists():
            return False
        age = _time.time() - p.stat().st_mtime
        return age >= 0 and age <= max_age_s
    except Exception:
        return False


def _set_wait_sched_debug(field: str, payload: dict) -> None:
    try:
        if field in _LAST_WAIT_SCHED_DEBUG:
            _LAST_WAIT_SCHED_DEBUG.pop(field, None)
        _LAST_WAIT_SCHED_DEBUG[field] = payload
        while len(_LAST_WAIT_SCHED_DEBUG) > _MAX_WAIT_SCHED_DEBUG:
            _LAST_WAIT_SCHED_DEBUG.popitem(last=False)
    except Exception:
        pass


def _diag_count(key: str, inc: int = 1) -> None:
    try:
        _DIAG_STATS[key] = _DIAG_STATS.get(key, 0) + inc
    except Exception:
        pass


def _dump_diag_stats() -> None:
    if os.environ.get("NAUTICAL_DIAG") == "1":
        try:
            elapsed = _ptime.perf_counter() - _DIAG_START_TS
            _DIAG_STATS["hook_seconds"] = round(elapsed, 4)
            _DIAG_STATS["run_task_seconds"] = round(_DIAG_STATS.get("run_task_seconds", 0.0), 4)
            parts = [f"{k}={v}" for k, v in _DIAG_STATS.items()]
            sys.stderr.write("[nautical] diag stats: " + ", ".join(parts) + "\n")
        except Exception:
            pass


def _diag_summary() -> None:
    if os.environ.get("NAUTICAL_DIAG") != "1":
        return
    try:
        parts = [
            f"spawn_deferred={_DIAG_STATS.get('spawn_deferred', 0)}",
            f"queue_lock_failures={_DIAG_STATS.get('queue_lock_failures', 0)}",
        ]
        sys.stderr.write("[nautical] diag summary: " + ", ".join(parts) + "\n")
    except Exception:
        pass


atexit.register(_dump_diag_stats)


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
# Locate nautical_core (single fixed location: ~/.task)
# ------------------------------------------------------------------------------
HOOK_DIR = Path(__file__).resolve().parent
TW_DIR = HOOK_DIR.parent

def _trusted_core_base(default_base: Path) -> Path:
    raw = (os.environ.get("NAUTICAL_CORE_PATH") or "").strip()
    if not raw:
        return default_base
    try:
        cand = Path(raw).expanduser().resolve()
    except Exception:
        return default_base
    if (os.environ.get("NAUTICAL_TRUST_CORE_PATH") or "").strip().lower() in ("1", "true", "yes", "on"):
        return cand
    try:
        st = os.stat(cand)
        uid_fn = getattr(os, "getuid", None)
        if callable(uid_fn) and st.st_uid != uid_fn():
            raise PermissionError("owner mismatch")
        if (st.st_mode & 0o002) != 0:
            raise PermissionError("path is world-writable")
        return cand
    except Exception as e:
        if os.environ.get("NAUTICAL_DIAG") == "1":
            try:
                sys.stderr.write(f"[nautical] Ignoring unsafe NAUTICAL_CORE_PATH '{raw}': {e}\n")
            except Exception:
                pass
        return default_base

_CORE_BASE = _trusted_core_base(TW_DIR)


def _core_target_from_base(base: Path) -> Path | None:
    try:
        if base.is_file():
            if base.name == "nautical_core.py":
                return base
            if base.name == "__init__.py" and base.parent.name == "nautical_core":
                return base
            return None
    except Exception:
        return None
    pyfile = base / "nautical_core.py"
    pkgini = base / "nautical_core" / "__init__.py"
    return pyfile if pyfile.is_file() else pkgini if pkgini.is_file() else None

core = None
_CORE_READY = False
_CORE_IMPORT_ERROR: Exception | None = None
_CORE_IMPORT_TARGET: Path | None = None
_HOOK_SUPPORT = None
_HOOK_SUPPORT_LOAD_FAILED = False
_MODIFY_QUERIES = None
_MODIFY_QUERIES_LOAD_FAILED = False
try:
    target = _core_target_from_base(_CORE_BASE)
    _CORE_IMPORT_TARGET = target
    if target:
        spec = importlib.util.spec_from_file_location("nautical_core", target)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules["nautical_core"] = module
            spec.loader.exec_module(module)
            core = module
except Exception as e:
    core = None
    _CORE_IMPORT_ERROR = e

def _resolve_task_data_context() -> tuple[str, bool]:
    resolver = getattr(core, "resolve_task_data_context", None) if core is not None else None
    if not callable(resolver):
        if core is not None:
            raise RuntimeError("nautical_core.resolve_task_data_context is required")
        if _CORE_IMPORT_ERROR is not None:
            target = str(_CORE_IMPORT_TARGET or (_CORE_BASE / "nautical_core.py"))
            raise RuntimeError(
                f"Failed to import nautical_core from {target}: "
                f"{type(_CORE_IMPORT_ERROR).__name__}: {_CORE_IMPORT_ERROR}"
            ) from _CORE_IMPORT_ERROR
        raise ModuleNotFoundError(
            f"nautical_core.py not found. Expected in ~/.task or NAUTICAL_CORE_PATH. "
            f"(resolved base: {_CORE_BASE})"
        )
    data_dir, use_rc, _source = resolver(
        argv=sys.argv[1:],
        env=os.environ,
        tw_dir=str(TW_DIR),
    )
    return str(data_dir), bool(use_rc)


_TASKDATA_RAW, _USE_RC_DATA_LOCATION = _resolve_task_data_context()
TW_DATA_DIR = Path(_TASKDATA_RAW).expanduser()


def _hook_support_target_from_base(base: Path) -> Path | None:
    try:
        if base.is_file():
            if base.name == "__init__.py" and base.parent.name == "nautical_core":
                target = base.parent / "hook_support.py"
                return target if target.is_file() else None
            return None
    except Exception:
        return None
    target = base / "nautical_core" / "hook_support.py"
    return target if target.is_file() else None


def _load_hook_support():
    global _HOOK_SUPPORT, _HOOK_SUPPORT_LOAD_FAILED
    if _HOOK_SUPPORT is not None:
        return _HOOK_SUPPORT
    if _HOOK_SUPPORT_LOAD_FAILED:
        return None
    base = _CORE_IMPORT_TARGET or _CORE_BASE
    target = _hook_support_target_from_base(base)
    if not target:
        _HOOK_SUPPORT_LOAD_FAILED = True
        return None
    try:
        spec = importlib.util.spec_from_file_location("nautical_hook_support", target)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules["nautical_hook_support"] = module
            spec.loader.exec_module(module)
            _HOOK_SUPPORT = module
            return module
    except Exception:
        pass
    _HOOK_SUPPORT_LOAD_FAILED = True
    return None


def _modify_queries_target_from_base(base: Path) -> Path | None:
    try:
        if base.is_file():
            if base.name == "__init__.py" and base.parent.name == "nautical_core":
                target = base.parent / "modify_queries.py"
                return target if target.is_file() else None
            return None
    except Exception:
        return None
    target = base / "nautical_core" / "modify_queries.py"
    return target if target.is_file() else None


def _load_modify_queries():
    global _MODIFY_QUERIES, _MODIFY_QUERIES_LOAD_FAILED
    if _MODIFY_QUERIES is not None:
        return _MODIFY_QUERIES
    if _MODIFY_QUERIES_LOAD_FAILED:
        return None
    base = _CORE_IMPORT_TARGET or _CORE_BASE
    target = _modify_queries_target_from_base(base)
    if not target:
        _MODIFY_QUERIES_LOAD_FAILED = True
        return None
    try:
        spec = importlib.util.spec_from_file_location("nautical_modify_queries", target)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules["nautical_modify_queries"] = module
            spec.loader.exec_module(module)
            _MODIFY_QUERIES = module
            return module
    except Exception:
        pass
    _MODIFY_QUERIES_LOAD_FAILED = True
    return None


def _task_cmd_prefix() -> list[str]:
    hook_support = _load_hook_support()
    if hook_support is not None:
        return hook_support.build_task_cmd_prefix(
            use_rc_data_location=_USE_RC_DATA_LOCATION,
            tw_data_dir=TW_DATA_DIR,
        )
    cmd = ["task"]
    if _USE_RC_DATA_LOCATION:
        cmd.append(f"rc.data.location={TW_DATA_DIR}")
    return cmd

# ------------------------------------------------------------------------------
# Deferred next-link spawn queue (used when nested `task import` times out due to TW lock)
# ------------------------------------------------------------------------------
_SPAWN_QUEUE_PATH = TW_DATA_DIR / ".nautical_spawn_queue.jsonl"
_SPAWN_QUEUE_LOCK = TW_DATA_DIR / ".nautical_spawn_queue.lock"
_SPAWN_QUEUE_DB_PATH = TW_DATA_DIR / ".nautical_queue.db"
_DEAD_LETTER_PATH = TW_DATA_DIR / ".nautical_dead_letter.jsonl"
_DEAD_LETTER_LOCK = TW_DATA_DIR / ".nautical_dead_letter.lock"
_SPAWN_LOCK_RETRIES = 6
_SPAWN_LOCK_SLEEP_BASE = 0.03
_SPAWN_LOCK_STALE_AFTER = 30.0
_DEAD_LETTER_LOCK_RETRIES = _SPAWN_LOCK_RETRIES
_DEAD_LETTER_LOCK_SLEEP_BASE = _SPAWN_LOCK_SLEEP_BASE
_WARNED_SPAWN_QUEUE_LOCK = False
_DURABLE_QUEUE = os.environ.get("NAUTICAL_DURABLE_QUEUE") == "1"

def _load_core() -> None:
    global core, _MAX_JSON_BYTES, _CORE_READY
    if core is not None and _CORE_READY:
        return
    if core is None:
        base = _CORE_BASE
        target = _core_target_from_base(base)
        if target:
            spec = importlib.util.spec_from_file_location("nautical_core", target)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules["nautical_core"] = module
                spec.loader.exec_module(module)
                core = module
    if core is None:
        msg = (
            "nautical_core.py not found. Expected in ~/.task or NAUTICAL_CORE_PATH. "
            f"(resolved base: {_CORE_BASE})"
        )
        raise ModuleNotFoundError(msg)
    try:
        core._warn_once_per_day_any("core_path", f"[nautical] core loaded: {getattr(core, '__file__', 'unknown')}")
    except Exception:
        pass
    try:
        _MAX_JSON_BYTES = int(getattr(core, "MAX_JSON_BYTES", _MAX_JSON_BYTES))
    except Exception:
        pass
    _apply_core_config()
    _CORE_READY = True

def _require_core() -> bool:
    try:
        _load_core()
        return core is not None
    except Exception:
        return False

# ------------------------------------------------------------------------------
# Config-driven toggles (env overrides still supported via core config helpers)
# ------------------------------------------------------------------------------
_CHAIN_COLOR_PER_CHAIN = True
_SHOW_TIMELINE_GAPS = False
_SHOW_ANALYTICS = False
_ANALYTICS_STYLE = "compact"
_ANALYTICS_ONTIME_TOL_SECS = 3600
_CHECK_CHAIN_INTEGRITY = False
_VERIFY_IMPORT = True
_DEBUG_WAIT_SCHED = _DEFAULT_DEBUG_WAIT_SCHED
_RECURRENCE_UPDATE_UDAS: tuple[str, ...] = ()
_SPAWN_QUEUE_MAX_BYTES = _DEFAULT_SPAWN_QUEUE_MAX_BYTES
_SPAWN_QUEUE_DRAIN_MAX_ITEMS = _SPAWN_QUEUE_DRAIN_MAX_ITEMS
_MAX_CHAIN_WALK = _MAX_CHAIN_WALK

def _apply_core_config() -> None:
    global _CHAIN_COLOR_PER_CHAIN, _SHOW_TIMELINE_GAPS, _SHOW_ANALYTICS, _ANALYTICS_STYLE
    global _ANALYTICS_ONTIME_TOL_SECS, _CHECK_CHAIN_INTEGRITY, _VERIFY_IMPORT
    global _DEBUG_WAIT_SCHED, _RECURRENCE_UPDATE_UDAS, _SPAWN_QUEUE_MAX_BYTES, _SPAWN_QUEUE_DRAIN_MAX_ITEMS, _MAX_CHAIN_WALK
    if core is None:
        return
    _CHAIN_COLOR_PER_CHAIN = core.CHAIN_COLOR_PER_CHAIN
    _SHOW_TIMELINE_GAPS = core.SHOW_TIMELINE_GAPS
    _SHOW_ANALYTICS = core.SHOW_ANALYTICS
    _ANALYTICS_STYLE = core.ANALYTICS_STYLE
    _ANALYTICS_ONTIME_TOL_SECS = core.ANALYTICS_ONTIME_TOL_SECS
    _CHECK_CHAIN_INTEGRITY = core.CHECK_CHAIN_INTEGRITY
    _VERIFY_IMPORT = core.VERIFY_IMPORT
    _DEBUG_WAIT_SCHED = core.DEBUG_WAIT_SCHED if hasattr(core, "DEBUG_WAIT_SCHED") else _DEFAULT_DEBUG_WAIT_SCHED
    _RECURRENCE_UPDATE_UDAS = tuple(core.RECURRENCE_UPDATE_UDAS) if hasattr(core, "RECURRENCE_UPDATE_UDAS") else ()
    _SPAWN_QUEUE_MAX_BYTES = core.SPAWN_QUEUE_MAX_BYTES if hasattr(core, "SPAWN_QUEUE_MAX_BYTES") else _DEFAULT_SPAWN_QUEUE_MAX_BYTES
    _SPAWN_QUEUE_DRAIN_MAX_ITEMS = core.SPAWN_QUEUE_DRAIN_MAX_ITEMS
    _MAX_CHAIN_WALK = core.MAX_CHAIN_WALK
_CHAIN_EXPORT_TIMEOUT_BASE = _env_float("NAUTICAL_CHAIN_EXPORT_TIMEOUT_BASE", _DEFAULT_CHAIN_EXPORT_TIMEOUT_BASE)
_CHAIN_EXPORT_TIMEOUT_PER_100 = _env_float("NAUTICAL_CHAIN_EXPORT_TIMEOUT_PER_100", _DEFAULT_CHAIN_EXPORT_TIMEOUT_PER_100)
_CHAIN_EXPORT_TIMEOUT_MAX = _env_float("NAUTICAL_CHAIN_EXPORT_TIMEOUT_MAX", _DEFAULT_CHAIN_EXPORT_TIMEOUT_MAX)
_CHAIN_EXPORT_TIMES: list[float] = []
_CHAIN_EXPORT_TIMES_MAX = 20
_CHAIN_EXPORT_TIMEOUT_FLOOR = _CHAIN_EXPORT_TIMEOUT_BASE


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
def _validate_anchor_expr_cached(expr: str) -> list[list[dict]]:
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
def _fail_and_exit(title: str, msg: str) -> None:
    _panel(f"❌ {title}", [("Message", msg)], kind="error")
    sys.exit(1)

_RAW_INPUT_TEXT = ""
_PARSED_NEW = None


def _fail_protocol_error(msg: str) -> None:
    _fail_and_exit("Protocol error", msg)


def _fail_invalid_input(msg: str) -> None:
    _fail_and_exit("Invalid input", msg)


def _task_uuid_or_empty(task: dict) -> str:
    if not isinstance(task, dict):
        return ""
    try:
        return str(task.get("uuid") or "").strip()
    except Exception:
        return ""


def _validate_modify_pair(old: dict, new: dict) -> tuple[dict, dict]:
    old_uuid = _task_uuid_or_empty(old)
    new_uuid = _task_uuid_or_empty(new)
    if not old_uuid or not new_uuid or old_uuid != new_uuid:
        _fail_protocol_error("Old and new task UUIDs differ")
    return old, new


def _validate_single_modify_task(task: dict) -> tuple[dict, dict]:
    if not _task_uuid_or_empty(task):
        _fail_protocol_error("Missing task UUID in on-modify input")
    return task, task


def _decode_leading_json_objects(raw: str, max_objects: int = 2) -> tuple[list[object], int]:
    decoder = json.JSONDecoder()
    idx = 0
    objs: list[object] = []
    n = len(raw)
    tries = 0
    loop_guard = 0
    max_loops = 10

    while idx < n and len(objs) < max_objects:
        loop_guard += 1
        if loop_guard > max_loops:
            _fail_protocol_error("Invalid JSON input: too many parse attempts")
        while idx < n and raw[idx].isspace():
            idx += 1
        if idx >= n:
            break
        try:
            obj, end = decoder.raw_decode(raw, idx)
        except Exception as e:
            _diag(f"json decode error: {e}")
            _fail_protocol_error("Invalid JSON input")
        objs.append(obj)
        if end <= idx:
            tries += 1
            if tries >= 2:
                _fail_protocol_error("Invalid JSON input: parser made no progress")
            idx += 1
            continue
        idx = end

    return objs, idx


def _read_two():
    global _RAW_INPUT_TEXT, _PARSED_NEW
    raw_bytes = sys.stdin.buffer.read(_MAX_JSON_BYTES + 1)
    if len(raw_bytes) > _MAX_JSON_BYTES:
        _fail_invalid_input(f"on-modify input exceeds {_MAX_JSON_BYTES} bytes")
    raw = raw_bytes.decode("utf-8", errors="replace")
    _RAW_INPUT_TEXT = raw
    if not raw or not raw.strip():
        _fail_invalid_input("on-modify must receive two JSON tasks")

    objs, idx = _decode_leading_json_objects(raw, max_objects=2)

    if len(objs) == 1 and isinstance(objs[0], list):
        if raw[idx:].strip():
            _fail_protocol_error("Invalid JSON input: trailing content")
        arr = [o for o in objs[0] if isinstance(o, dict)]
        if len(arr) >= 2:
            _PARSED_NEW = arr[-1]
            old, new = _validate_modify_pair(arr[0], arr[-1])
            return old, new
        if len(arr) == 1:
            _PARSED_NEW = arr[0]
            only, _ = _validate_single_modify_task(arr[0])
            return only, only

    objs = [o for o in objs if isinstance(o, dict)]
    if len(objs) >= 2:
        if raw[idx:].strip():
            _fail_protocol_error("Invalid JSON input: trailing content")
        _PARSED_NEW = objs[-1]
        old, new = _validate_modify_pair(objs[0], objs[-1])
        return old, new
    if len(objs) == 1:
        _PARSED_NEW = objs[0]
        only, _ = _validate_single_modify_task(objs[0])
        return only, only

    _fail_invalid_input("on-modify must receive two JSON tasks")


def _decode_latest_task_from_raw(raw: str) -> dict | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    decoder = json.JSONDecoder()
    idx = 0
    n = len(raw)
    last_task = None
    while idx < n:
        while idx < n and raw[idx].isspace():
            idx += 1
        if idx >= n:
            break
        try:
            obj, end = decoder.raw_decode(raw, idx)
        except Exception:
            break
        if isinstance(obj, dict):
            last_task = obj
        elif isinstance(obj, list):
            arr = [x for x in obj if isinstance(x, dict)]
            if arr:
                last_task = arr[-1]
        if end <= idx:
            break
        idx = end
    return last_task if isinstance(last_task, dict) else None


def _panic_passthrough() -> None:
    fallback: dict[str, Any] = {}
    task = _PARSED_NEW if isinstance(_PARSED_NEW, dict) else None
    if task is None:
        task = _decode_latest_task_from_raw(_RAW_INPUT_TEXT)
    try:
        print(json.dumps(task if isinstance(task, dict) else fallback, ensure_ascii=False), end="")
    except Exception:
        try:
            print("{}", end="")
        except Exception:
            pass
    try:
        sys.stdout.flush()
    except Exception:
        pass


def _task_has_nautical_fields(old: dict, new: dict) -> bool:
    keys = ("anchor", "anchor_mode", "cp", "chainID", "chainid", "nextLink", "prevLink", "link")
    for task in (old, new):
        if not isinstance(task, dict):
            continue
        for key in keys:
            val = task.get(key)
            if val is None:
                continue
            try:
                s = str(val).strip()
            except Exception:
                s = ""
            if s:
                return True
    return False


def _print_task(task):
    if core is None:
        try:
            _load_core()
        except Exception:
            print(json.dumps(task, ensure_ascii=False), end="")
            return
    if core.SANITIZE_UDA:
        core.sanitize_task_strings(task, max_len=core.SANITIZE_UDA_MAX_LEN)
    print(json.dumps(task, ensure_ascii=False), end="")




_PANEL_THEMES = {
    "preview_anchor": {"border": "turquoise2", "title": "bright_cyan", "label": "light_sea_green"},
    "preview_cp": {"border": "deep_pink1", "title": "deep_pink1", "label": "deep_pink3"},
    "summary": {"border": "indian_red", "title": "indian_red", "label": "red"},
    "disabled": {"border": "yellow", "title": "yellow", "label": "yellow"},
    "error": {"border": "red", "title": "red", "label": "red"},
    "warning": {"border": "yellow", "title": "yellow", "label": "yellow"},
    "info": {"border": "blue", "title": "cyan", "label": "cyan"},
}


def _panel(
    title,
    rows,
    kind: str = "info",
    border_style: str | None = None,
    title_style: str | None = None,
    label_style: str | None = None,
):
    if core is None:
        try:
            _load_core()
        except Exception:
            try:
                sys.stderr.write(f"[nautical] {title}\n")
            except Exception:
                pass
            return
    themes = dict(_PANEL_THEMES)
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
        fast_color=core.FAST_COLOR,
        themes=themes,
        allow_line=True,
        line_force_rich_kinds={"summary"},
        label_width_min=6,
        label_width_max=14,
    )


def _panel_line_from_rows(title, rows) -> str:
    return core.panel_line_from_rows(title, rows)


def _panel_line(
    title: str,
    line: str,
    *,
    kind: str = "info",
    border_style: str | None = None,
    title_style: str | None = None,
) -> None:
    core.panel_line(
        title,
        line,
        kind=kind,
        themes=_PANEL_THEMES,
        border_style=border_style,
        title_style=title_style,
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



def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_DIRECTORY)
    except Exception:
        return
    try:
        os.fsync(fd)
    except Exception:
        pass
    finally:
        try:
            os.close(fd)
        except Exception:
            pass


def _spawn_queue_db_size_bytes() -> int:
    total = 0
    paths = (
        _SPAWN_QUEUE_DB_PATH,
        Path(str(_SPAWN_QUEUE_DB_PATH) + "-wal"),
        Path(str(_SPAWN_QUEUE_DB_PATH) + "-shm"),
    )
    for p in paths:
        try:
            if p.exists():
                total += p.stat().st_size
        except Exception:
            continue
    return total


def _spawn_queue_total_bytes() -> int:
    return _spawn_queue_db_size_bytes()


def _spawn_queue_warn_growth(queue_path: Path, size: int) -> None:
    global _WARNED_SPAWN_QUEUE_GROWTH
    try:
        if _WARNED_SPAWN_QUEUE_GROWTH:
            return
        if _SPAWN_QUEUE_MAX_BYTES <= 0:
            return
        if size > _SPAWN_QUEUE_MAX_BYTES:
            _WARNED_SPAWN_QUEUE_GROWTH = True
            _panel(
                "⚠ Spawn queue growing",
                [
                    ("Queue", str(queue_path)),
                    ("Size", f"{size} bytes"),
                    ("Limit", f"{_SPAWN_QUEUE_MAX_BYTES} bytes"),
                    ("Hint", "Run the on-exit hook or reduce load."),
                ],
                kind="warning",
            )
    except Exception:
        pass


def _handle_enqueue_lock_busy(task_obj: dict) -> tuple[bool, str]:
    _write_dead_letter(task_obj, "queue lock busy")
    _diag("queue lock busy; intent dead-lettered")
    _diag_count("queue_lock_failures")
    global _WARNED_SPAWN_QUEUE_LOCK
    if not _WARNED_SPAWN_QUEUE_LOCK:
        _WARNED_SPAWN_QUEUE_LOCK = True
        _panel(
            "⚠ Spawn queue busy",
            [
                ("Queue", str(_SPAWN_QUEUE_DB_PATH)),
                ("Hint", "Queue lock busy; spawn intent not queued."),
            ],
            kind="warning",
        )
    return False, "queue lock busy"


def _spawn_queue_db_connect() -> sqlite3.Connection | None:
    try:
        _SPAWN_QUEUE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    try:
        connect_timeout = max(1.0, _SPAWN_LOCK_SLEEP_BASE * max(1, _SPAWN_LOCK_RETRIES) * 4.0)
        busy_timeout_ms = int(min(60_000, max(1_500, connect_timeout * 1000.0 * 2.0)))
        conn = sqlite3.connect(
            str(_SPAWN_QUEUE_DB_PATH),
            timeout=connect_timeout,
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
        try:
            if _SPAWN_QUEUE_DB_PATH.exists():
                os.chmod(_SPAWN_QUEUE_DB_PATH, 0o600)
        except Exception:
            pass
        return conn
    except Exception as e:
        _diag(f"queue db connect failed: {e}")
        return None


def _spawn_queue_db_init(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS queue_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            spawn_intent_id TEXT,
            payload TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            state TEXT NOT NULL DEFAULT 'queued',
            claim_token TEXT,
            claimed_at REAL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_queue_entries_spawn_intent
        ON queue_entries (spawn_intent_id)
        WHERE spawn_intent_id IS NOT NULL AND spawn_intent_id <> ''
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_queue_entries_state_id ON queue_entries (state, id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_queue_entries_claimed_at ON queue_entries (claimed_at)")
    conn.commit()


def _spawn_queue_capacity_guard(task_obj: dict) -> tuple[bool, str] | None:
    if _SPAWN_QUEUE_MAX_BYTES <= 0:
        return None
    try:
        if _spawn_queue_total_bytes() > _SPAWN_QUEUE_MAX_BYTES:
            _write_dead_letter(task_obj, "spawn queue full")
            _diag("spawn queue full; intent dropped")
            return False, "spawn queue full"
    except Exception:
        pass
    return None


def _spawn_queue_db_rollback(conn: sqlite3.Connection) -> None:
    try:
        conn.rollback()
    except Exception:
        pass


def _spawn_queue_payload(task_obj: dict) -> tuple[str, float, str]:
    payload = json.dumps(task_obj, ensure_ascii=False, separators=(",", ":"))
    now = _time.time()
    spawn_intent_id = (task_obj.get("spawn_intent_id") or "").strip() if isinstance(task_obj, dict) else ""
    return payload, now, spawn_intent_id


def _spawn_queue_upsert(
    conn: sqlite3.Connection,
    *,
    payload: str,
    now: float,
    spawn_intent_id: str,
) -> None:
    if spawn_intent_id:
        cur = conn.execute(
            "UPDATE queue_entries SET payload=?, state='queued', claim_token=NULL, claimed_at=NULL, updated_at=? "
            "WHERE spawn_intent_id=?",
            (payload, now, spawn_intent_id),
        )
        if int(getattr(cur, "rowcount", 0) or 0) > 0:
            return
        try:
            conn.execute(
                "INSERT INTO queue_entries (spawn_intent_id, payload, attempts, state, claim_token, claimed_at, created_at, updated_at) "
                "VALUES (?, ?, 0, 'queued', NULL, NULL, ?, ?)",
                (spawn_intent_id, payload, now, now),
            )
            return
        except sqlite3.IntegrityError:
            conn.execute(
                "UPDATE queue_entries SET payload=?, state='queued', claim_token=NULL, claimed_at=NULL, updated_at=? "
                "WHERE spawn_intent_id=?",
                (payload, now, spawn_intent_id),
            )
            return
    conn.execute(
        "INSERT INTO queue_entries (spawn_intent_id, payload, attempts, state, claim_token, claimed_at, created_at, updated_at) "
        "VALUES (NULL, ?, 0, 'queued', NULL, NULL, ?, ?)",
        (payload, now, now),
    )


def _spawn_queue_repair_permissions() -> None:
    try:
        if _SPAWN_QUEUE_DB_PATH.exists():
            os.chmod(_SPAWN_QUEUE_DB_PATH, 0o600)
        wal = Path(str(_SPAWN_QUEUE_DB_PATH) + "-wal")
        if wal.exists():
            os.chmod(wal, 0o600)
        shm = Path(str(_SPAWN_QUEUE_DB_PATH) + "-shm")
        if shm.exists():
            os.chmod(shm, 0o600)
    except Exception:
        pass


def _spawn_queue_write_failure(task_obj: dict, err: Exception) -> tuple[bool, str]:
    fail_reason = f"spawn queue write failed: {err}"
    _write_dead_letter(task_obj, fail_reason)
    _diag(fail_reason)
    return False, fail_reason


def _enqueue_deferred_spawn_sqlite(task_obj: dict) -> tuple[bool, str]:
    guard = _spawn_queue_capacity_guard(task_obj)
    if guard is not None:
        return guard

    conn = _spawn_queue_db_connect()
    if conn is None:
        return False, "spawn queue db unavailable"
    try:
        _spawn_queue_db_init(conn)
        payload, now, spawn_intent_id = _spawn_queue_payload(task_obj)
        conn.execute("BEGIN IMMEDIATE")
        _spawn_queue_upsert(
            conn,
            payload=payload,
            now=now,
            spawn_intent_id=spawn_intent_id,
        )
        conn.commit()
        _spawn_queue_repair_permissions()
        _spawn_queue_warn_growth(_SPAWN_QUEUE_DB_PATH, _spawn_queue_db_size_bytes())
        return True, ""
    except sqlite3.OperationalError as e:
        _spawn_queue_db_rollback(conn)
        msg = str(e).lower()
        if "locked" in msg or "busy" in msg:
            return _handle_enqueue_lock_busy(task_obj)
        return _spawn_queue_write_failure(task_obj, e)
    except Exception as e:
        _spawn_queue_db_rollback(conn)
        return _spawn_queue_write_failure(task_obj, e)
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _enqueue_deferred_spawn(task_obj: dict) -> tuple[bool, str]:
    return _enqueue_deferred_spawn_sqlite(task_obj)


def _write_dead_letter(entry: dict, reason: str) -> None:
    if not _require_core():
        return
    payload = {
        "ts": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
        "hook": "on-modify",
        "hook_version": NAUTICAL_HOOK_VERSION,
        "reason": reason,
        "spawn_intent_id": (entry.get("spawn_intent_id") or "").strip(),
        "parent_uuid": (entry.get("parent_uuid") or "").strip(),
        "child_short": (entry.get("child_short") or "").strip(),
        "child_uuid": ((entry.get("child") or {}).get("uuid") or "").strip(),
        "entry": entry,
    }
    try:
        _DEAD_LETTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    try:
        with core.safe_lock(
            _DEAD_LETTER_LOCK,
            retries=_DEAD_LETTER_LOCK_RETRIES,
            sleep_base=_DEAD_LETTER_LOCK_SLEEP_BASE,
            jitter=_DEAD_LETTER_LOCK_SLEEP_BASE,
            mkdir=True,
            stale_after=_SPAWN_LOCK_STALE_AFTER,
        ) as ok:
            if not ok:
                _diag("dead-letter lock busy; entry not recorded")
                return
            fd = os.open(str(_DEAD_LETTER_PATH), os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
            try:
                os.fchmod(fd, 0o600)
            except Exception:
                pass
            with os.fdopen(fd, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
                if _DURABLE_QUEUE:
                    try:
                        f.flush()
                        os.fsync(f.fileno())
                        _fsync_dir(_DEAD_LETTER_PATH.parent)
                    except Exception:
                        pass
    except Exception:
        pass



def _short(u):
    return (u or "")[:8]


def _run_task(
    cmd: list[str],
    *,
    env: dict | None = None,
    input_text: str | None = None,
    timeout: float = 3.0,
    retries: int = 2,
    retry_delay: float = 0.15,
    use_tempfiles: bool = False,
) -> tuple[bool, str, str]:
    load_err = None
    try:
        _load_core()
    except Exception as e:
        load_err = e
    _diag_count("run_task_calls")
    t0 = _ptime.perf_counter()
    core_runner = getattr(core, "run_task", None) if core is not None else None
    hook_support = _load_hook_support()
    if hook_support is not None:
        if load_err is not None and not callable(core_runner):
            _diag(f"core.run_task unavailable; falling back to subprocess: {load_err}")
        ok, out, err = hook_support.run_task(
            cmd,
            core_run_task=core_runner,
            env=env,
            input_text=input_text,
            timeout=timeout,
            retries=retries,
            retry_delay=retry_delay,
            use_tempfiles=use_tempfiles,
        )
    else:
        if callable(core_runner):
            ok, out, err = core.run_task(
                cmd,
                env=env,
                input_text=input_text,
                timeout=timeout,
                retries=retries,
                retry_delay=retry_delay,
                use_tempfiles=use_tempfiles,
            )
        else:
            if load_err is not None:
                _diag(f"core.run_task unavailable; falling back to subprocess: {load_err}")
            env = env or os.environ.copy()
            proc = None
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    close_fds=True,
                    env=env,
                )
                out, err = proc.communicate(input=input_text, timeout=timeout)
                ok, out, err = (proc.returncode == 0, out or "", err or "")
            except subprocess.TimeoutExpired:
                if proc is not None:
                    proc.kill()
                try:
                    out, err = proc.communicate(timeout=1.0) if proc is not None else ("", "")
                except Exception:
                    out, err = "", ""
                ok, out, err = (False, out or "", "timeout")
            except Exception as e:
                if proc is not None:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    try:
                        proc.wait(timeout=1.0)
                    except Exception:
                        pass
                ok, out, err = (False, "", str(e))
    _diag_count("run_task_seconds", _ptime.perf_counter() - t0)
    if not ok:
        _diag_count("run_task_failures")
    return ok, out, err


def _export_uuid_short(u_short: str, env=None):
    cache_chain_id = ""
    if env is None and u_short:
        with _CHAIN_CACHE_LOCK:
            cached = _CHAIN_BY_SHORT.get(u_short)
            cache_chain_id = _CHAIN_CACHE_CHAIN_ID
        if isinstance(cached, dict):
            _diag_count("export_uuid_cache_hits")
            return dict(cached)
    if env is None:
        if cache_chain_id:
            _diag_count("unexpected_cache_misses")
            _diag(f"cache miss: short uuid {u_short} (chainID={cache_chain_id})")
        else:
            _diag_count("export_uuid_cache_misses")
    hook_support = _load_hook_support()
    if hook_support is not None:
        return hook_support.export_uuid_short(
            run_task=_run_task,
            task_cmd_prefix=_task_cmd_prefix(),
            uuid_short=u_short,
            env=(env or os.environ.copy()),
            timeout=2.5,
            retries=2,
            diag=_diag,
        )
    env = env or os.environ.copy()
    ok, out, err = _run_task(
        _task_cmd_prefix() + ["rc.hooks=off", "rc.json.array=off", f"uuid:{u_short}", "export"],
        env=env,
        timeout=2.5,
        retries=2,
    )
    if not ok:
        _diag(f"export uuid:{u_short} failed: {err.strip()}")
        return None
    try:
        obj = json.loads(out.strip() or "{}")
        if not obj.get("uuid"):
            return None
        if not str(obj.get("uuid") or "").lower().startswith((u_short or "").lower()):
            _diag(f"uuid prefix mismatch for {u_short}")
            return None
        return obj
    except Exception:
        return None


def _task_exists_by_uuid(u: str, env: dict | None) -> bool:
    if env is None:
        return _task_exists_by_uuid_cached(u)
    return _task_exists_by_uuid_uncached(u, env=env)


@lru_cache(maxsize=512)
def _task_exists_by_uuid_cached(u: str) -> bool:
    return _task_exists_by_uuid_uncached(u, env=None)


def _task_exists_by_uuid_uncached(u: str, env: dict | None) -> bool:
    hook_support = _load_hook_support()
    if hook_support is not None:
        return hook_support.task_exists_by_uuid_uncached(
            run_task=_run_task,
            task_cmd_prefix=_task_cmd_prefix(),
            uuid_str=u,
            env=env,
            timeout=2.5,
            retries=2,
            diag=_diag,
        )
    q = _task_cmd_prefix() + ["rc.hooks=off", "rc.json.array=off", f"uuid:{u}", "export"]
    ok, out, err = _run_task(q, env=env, timeout=2.5, retries=2)
    if not ok:
        _diag(f"task exists check failed (uuid={u[:8]}): {err.strip()}")
        return False
    try:
        data = json.loads(out.strip() or "{}")
    except Exception:
        data = {}
    return bool(data.get("uuid"))

def _reserve_child_uuid(env: dict) -> str:
    candidate = str(uuid.uuid4())
    while True:
        ok, out, err = _run_task(
            _task_cmd_prefix() + ["rc.hooks=off", "rc.json.array=off", f"uuid:{candidate}", "count"],
            env=env,
            timeout=2.5,
            retries=2,
        )
        if ok:
            if (out or "").strip() == "0":
                return candidate
            candidate = str(uuid.uuid4())
            continue
        _diag(f"uuid availability check failed (uuid={candidate[:8]}): {err.strip()}")
        return candidate


def _stable_child_uuid(parent_task: dict | None, child_task: dict | None) -> str:
    """Return a cross-device-stable UUID for a child slot when possible."""
    if not isinstance(parent_task, dict) or not isinstance(child_task, dict):
        return ""
    parent_uuid = _task_uuid_or_empty(parent_task)
    if not parent_uuid:
        return ""
    link_no = core.coerce_int(child_task.get("link"), None)
    if link_no is None:
        return ""
    chain_id = (
        child_task.get("chainID")
        or child_task.get("chainid")
        or parent_task.get("chainID")
        or parent_task.get("chainid")
        or ""
    )
    kind = "anchor" if (parent_task.get("anchor") or "").strip() else "cp" if (parent_task.get("cp") or "").strip() else ""
    slot_key = json.dumps(
        {
            "chain_id": str(chain_id).strip().lower(),
            "kind": kind,
            "link": int(link_no),
            "parent_uuid": parent_uuid.lower(),
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return str(uuid.uuid5(_STABLE_CHILD_UUID_NAMESPACE, slot_key))


def _child_uuid_for_spawn(parent_task: dict | None, child_task: dict | None, env: dict) -> str:
    stable = _stable_child_uuid(parent_task, child_task)
    if stable:
        return stable
    return _reserve_child_uuid(env)


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

def _emit_line(msg: str) -> None:
    if not msg:
        return
    try:
        sys.stderr.write(msg + "\n")
    except Exception:
        pass

def _format_line_cap(base_no: int, cap_no: int | None, until_dt: datetime | None, until_no: int | None) -> str:
    parts = []
    if cap_no:
        left = max(0, cap_no - base_no)
        parts.append(f"cap {cap_no}, left {left}")
    if until_dt:
        until_txt = core.fmt_dt_local(until_dt)
        parts.append(f"until {until_txt}")
    return f" ({'; '.join(parts)})" if parts else ""

def _format_line_preview(
    link_no: int,
    task: dict,
    child_due_utc: datetime,
    child_short: str,
    now_utc: datetime,
    cap_no: int | None = None,
    until_dt: datetime | None = None,
    until_no: int | None = None,
) -> str:
    cur_due = _dtparse(task.get("due"))
    cur_end = _dtparse(task.get("end"))
    delta_txt = core.strip_rich_markup(_fmt_on_time_delta(cur_due, cur_end) or "").strip()
    if delta_txt.startswith("(") and delta_txt.endswith(")"):
        delta_txt = delta_txt[1:-1].strip()
    if delta_txt:
        if not delta_txt.endswith(","):
            delta_txt = f"{delta_txt},"
    due_local = core.fmt_dt_local(child_due_utc) if child_due_utc else "—"
    due_delta = _human_delta(now_utc, child_due_utc, False)
    if due_delta.startswith("in "):
        due_delta = "due " + due_delta
    elif due_delta.startswith("overdue by "):
        due_delta = due_delta
    else:
        due_delta = "due " + due_delta
    cap_txt = _format_line_cap(link_no, cap_no, until_dt, until_no)
    parts = [str(link_no), "✓"]
    if delta_txt:
        parts.append(delta_txt)
    parts.append("next")
    parts.append("►")
    parts.append(due_local)
    parts.append(due_delta)
    line = " ".join(p for p in parts if p)
    if cap_txt:
        line = line + cap_txt
    return line.strip()


def _spawn_child(child_task: dict, parent_task: dict | None = None) -> tuple[str, set[str]]:
    """
    Create child via `task import -`, preserving annotation entries.
    Returns (short_uuid, stripped_attrs).
    Raises RuntimeError with detailed context on failure.
    """
    env = os.environ.copy()
    child_uuid = _child_uuid_for_spawn(parent_task, child_task, env)
    obj = dict(child_task)
    obj["uuid"] = child_uuid
    if "entry" not in obj:
        obj["entry"] = core.fmt_isoz(core.now_utc())
    if "modified" not in obj:
        obj["modified"] = obj["entry"]


    obj = _strip_none_and_cast(obj)
    _normalise_datetime_fields(obj)

    attempts = 0
    stripped_accum: set[str] = set()
    last_stderr = ""
    last_category = ""

    while attempts < _MAX_SPAWN_ATTEMPTS:
        attempts += 1
        payload = json.dumps(obj) + "\n"

        ok, _out, err = _run_task(
            _task_cmd_prefix() + ["rc.hooks=off", "import", "-"],
            input_text=payload,
            env=env,
            timeout=10,  # prevent hanging
            retries=1,
        )
        if not ok and err == "timeout":
            last_stderr = "Task import timed out (>10s)"
            last_category = "taskwarrior"
            continue

        if ok:
            # Always verify existence to avoid reporting success on partial import failures.
            if _task_exists_by_uuid(child_uuid, env):
                return child_uuid[:8], stripped_accum
            if not _VERIFY_IMPORT:
                _diag("verify_import=false configured, but strict child existence verification is enforced")
            last_stderr = "task import reported success but child task was not found by UUID"
            category, is_retryable = ("taskwarrior", True)
        else:
            last_stderr = err or ""
            category, is_retryable = _categorize_spawn_error(1, last_stderr)
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


def _spawn_intent_entry(
    parent_uuid: str,
    child_obj: dict,
    child_short: str,
    parent_nextlink: str | None = None,
    spawn_intent_id: str | None = None,
) -> dict:
    intent_id = (spawn_intent_id or "").strip()
    if not intent_id:
        intent_id = f"si_{uuid.uuid4().hex[:12]}"
    return {
        "parent_uuid": parent_uuid,
        "parent_nextlink": (parent_nextlink or "").strip(),
        "child_short": child_short,
        "child": child_obj,
        "spawn_intent_id": intent_id,
    }


def _enqueue_spawn_intent(entry: dict) -> tuple[bool, str]:
    if not isinstance(entry, dict):
        return False, "invalid spawn intent"
    return _enqueue_deferred_spawn(entry)


def _spawn_child_atomic(
    child_task: dict,
    parent_task_with_nextlink: dict,
) -> tuple[str, set[str], bool, bool, str | None, str | None]:
    """
    Queue a child spawn intent for the on-exit hook.

    Important: The parent update is applied by Taskwarrior using this hook's stdout.
    We intentionally avoid importing the parent from inside the hook to reduce the
    risk of re-entering Taskwarrior while it is holding the datastore lock.

    We enqueue the child for the on-exit hook to import, then update the parent link.
    """
    env = os.environ.copy()

    # Prepare child with a stable UUID so retries and cross-device completions converge.
    child_uuid = _child_uuid_for_spawn(parent_task_with_nextlink, child_task, env)
    spawn_intent_id = f"si_{uuid.uuid4().hex[:12]}"
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

    # Decision-only mode: enqueue for on-exit spawn and return unverified.
    entry = _spawn_intent_entry(
        parent_task_with_nextlink.get("uuid") or "",
        child_obj,
        child_short,
        parent_task_with_nextlink.get("nextLink") or "",
        spawn_intent_id,
    )
    queued, queue_reason = _enqueue_spawn_intent(entry)
    if not queued:
        return (
            child_short,
            stripped_attrs,
            False,
            False,
            f"Spawn intent queue failed: {queue_reason}",
            spawn_intent_id,
        )
    _diag_count("spawn_deferred")
    return (
        child_short,
        stripped_attrs,
        False,
        True,
        "Spawn intent queued for on-exit processing",
        spawn_intent_id,
    )



def _root_uuid_from(task: dict) -> str:
    """Return the stable chain seed.

    ChainID is the only source of truth.
    """
    return (task.get("chainID") or task.get("chainid") or "").strip()

# --- Chain export: chainID is mandatory --------------------------------------
def _task(args, env=None) -> str:
    """
    Thin wrapper around 'task' returning stdout as text.
    Always disables hooks; caller should provide rc.json.array flag when needed.
    """
    modify_queries = _load_modify_queries()
    if modify_queries is not None:
        return modify_queries.task_text(
            args,
            run_task=_run_task,
            task_cmd_prefix=_task_cmd_prefix(),
            env=(env or os.environ.copy()),
            timeout=3.0,
            retries=2,
            diag=_diag,
        )
    env = env or os.environ.copy()
    ok, out, err = _run_task(
        _task_cmd_prefix() + ["rc.hooks=off"] + list(args),
        env=env,
        timeout=3.0,
        retries=2,
    )
    if not ok:
        _diag(f"task {' '.join(args)} failed: {err.strip()}")
    return out or ""

def _export_uuid_full(u: str, env=None) -> dict | None:
    """Export a single task by full UUID."""
    if env is None:
        res = _export_uuid_full_cached(u)
        return dict(res) if isinstance(res, dict) else None
    return _export_uuid_full_uncached(u, env=env)


@lru_cache(maxsize=256)
def _export_uuid_full_cached(u: str) -> dict | None:
    return _export_uuid_full_uncached(u, env=None)


def _export_uuid_full_uncached(u: str, env=None) -> dict | None:
    cache_chain_id = ""
    if env is None and u:
        with _CHAIN_CACHE_LOCK:
            cached = _CHAIN_BY_UUID.get(u)
            cache_chain_id = _CHAIN_CACHE_CHAIN_ID
        if isinstance(cached, dict):
            _diag_count("export_full_cache_hits")
            return dict(cached)
    if env is None:
        if cache_chain_id:
            _diag_count("unexpected_cache_misses")
            _diag(f"cache miss: full uuid {u} (chainID={cache_chain_id})")
        else:
            _diag_count("export_full_cache_misses")
    hook_support = _load_hook_support()
    if hook_support is not None:
        return hook_support.export_uuid_full(
            run_task=_run_task,
            task_cmd_prefix=_task_cmd_prefix(),
            uuid_str=u,
            env=env,
            timeout=3.0,
            retries=2,
            diag=_diag,
        )
    try:
        out = _task(["rc.json.array=1", f"export uuid:{u}"], env=env)  # uses existing _task()
        arr = json.loads(out) if out and out.strip().startswith("[") else []
        return arr[0] if arr else None
    except Exception:
        return None

def tw_export_chain_required(seed_task, env=None):
    """Return full chain export for a task.

    Policy: chainID is mandatory.
    """
    chain_id = seed_task.get('chainID') or seed_task.get('chainid')
    if not chain_id:
        raise RuntimeError(
            "ChainID is required (legacy chain traversal removed). "
            "Run tools/nautical_backfill_chainid.py, then retry."
        )
    if env is None:
        return _get_chain_export(chain_id)
    return tw_export_chain(chain_id, env=env, limit=_MAX_CHAIN_WALK)
def _tw_get_cached(ref: str) -> str:
    """Return `task _get <ref>` stdout stripped. Cached within one hook run."""
    try:
        if ref.endswith(".entry"):
            short = ref[:-6].strip()
            with _CHAIN_CACHE_LOCK:
                cached = _CHAIN_BY_SHORT.get(short) if short else None
                cache_chain_id = _CHAIN_CACHE_CHAIN_ID
            if short and isinstance(cached, dict):
                _diag_count("tw_get_cache_hits")
                return (str(cached.get("entry") or "")).strip()
            if short and cache_chain_id:
                _diag_count("unexpected_cache_misses")
                _diag(f"cache miss: _get {ref} (chainID={cache_chain_id})")
        modify_queries = _load_modify_queries()
        if modify_queries is not None:
            return modify_queries.tw_get(
                ref,
                task_text=lambda args: _task(args, env=None),
            )
        out = _task(["rc.verbose=nothing", "_get", ref], env=None)
        return (out or "").strip()
    except Exception:
        return ""

def _chain_root_and_age(task: dict, now_utc: datetime) -> tuple[str, int | None]:
    """Get chain root (chainID) and age in days.
    Returns (root_short, age_days). age_days is None if unavailable."""
    modify_queries = _load_modify_queries()
    if modify_queries is not None:
        return modify_queries.chain_root_and_age(
            task,
            now_utc,
            root_uuid_from=_root_uuid_from,
            tw_get_cached=_tw_get_cached,
            dtparse=_dtparse,
            tolocal=_tolocal,
        )
    try:
        root_short = _root_uuid_from(task)
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
        return root_short or "—", age_days
    except Exception:
        return "—", None

def _format_root_and_age(task: dict, now_utc: datetime) -> str:
    """Format root and age as a single string.
    Returns root (age) or just root if age is 0 or unavailable."""
    modify_queries = _load_modify_queries()
    if modify_queries is not None:
        return modify_queries.format_root_and_age(
            task,
            now_utc,
            chain_root_and_age=_chain_root_and_age,
        )
    root_short, age_days = _chain_root_and_age(task, now_utc)
    if not root_short or root_short == "—":
        return "—"
    if age_days is not None and age_days > 0:
        return f"{root_short} ▻ {age_days}d"
    return root_short

# ------------------------------------------------------------------------------
# On modify-without-completion helpers
# ------------------------------------------------------------------------------


def _canon_for_compare(v):
    """Canonicalize values so 5 == 5.0, strings are trimmed, and
    dict/list comparisons are stable."""
    from decimal import Decimal, InvalidOperation
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
    root = _root_uuid_from(task)
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


def _collect_prev_two(current_task: dict, chain_by_link: dict[int, list[dict]] | None = None) -> list[dict]:
    """Return up to two previous tasks (older first) using chainID export only."""

    chain_id = (current_task.get("chainID") or "").strip()
    if not chain_id:
        return []

    cur_no = core.coerce_int(current_task.get("link"), None)
    if not cur_no or cur_no <= 1:
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

    if chain_by_link is None:
        if _PANEL_CHAIN_BY_LINK:
            chain_by_link = _PANEL_CHAIN_BY_LINK
        else:
            chain_by_link = {}
    if not chain_by_link:
        try:
            chain = _get_chain_export(chain_id)
        except Exception:
            return []
        for t in chain:
            ln = core.coerce_int(t.get("link"), None)
            if ln is None:
                continue
            chain_by_link.setdefault(ln, []).append(t)

    prevs: list[dict] = []
    for want in (cur_no - 2, cur_no - 1):
        if want < 1:
            continue
        obj = _pick_best(chain_by_link.get(want, []))
        if obj:
            prevs.append(obj)
    return prevs


@lru_cache(maxsize=32)
def _tw_export_chain_cached_key(chain_id: str, since_key: str, extra_key: str, limit: int) -> tuple[dict, ...]:
    """Cached chain export keyed by stable parameters."""
    since = datetime.fromisoformat(since_key) if since_key else None
    extra = extra_key or None
    with _CHAIN_CACHE_LOCK:
        if _CHAIN_CACHE_CHAIN_ID and chain_id == _CHAIN_CACHE_CHAIN_ID and not since and not extra:
            return tuple(_CHAIN_CACHE or [])
    return tuple(tw_export_chain(chain_id, since=since, extra=extra, env=None, limit=limit) or [])


def _tw_export_chain_cached(chain_id: str, since: datetime | None, extra: str | None, limit: int) -> tuple[dict, ...]:
    since_key = since.isoformat() if isinstance(since, datetime) else ""
    extra_key = str(extra or "")
    return _tw_export_chain_cached_key(chain_id, since_key, extra_key, limit)


def _get_chain_export(chain_id: str, since: datetime | None = None, extra: str | None = None, env=None) -> list[dict]:
    """Return a safe list copy of a chain export (cached when env is None)."""
    if not chain_id:
        return []
    if env is not None:
        return tw_export_chain(chain_id, since=since, extra=extra, env=env, limit=_MAX_CHAIN_WALK)
    with _CHAIN_CACHE_LOCK:
        if _CHAIN_CACHE_CHAIN_ID and chain_id == _CHAIN_CACHE_CHAIN_ID and not since and not extra:
            return list(_CHAIN_CACHE)
    cached = _tw_export_chain_cached(chain_id, since, extra, _MAX_CHAIN_WALK)
    return list(cached)


def _existing_next_task(parent_task: dict, next_no: int) -> dict | None:
    """Return an existing next-link task for idempotent re-completion handling."""
    if not isinstance(parent_task, dict):
        return None

    # Fast path: explicit nextLink pointer on parent.
    next_ref = (parent_task.get("nextLink") or "").strip()
    if next_ref:
        obj = _export_uuid_short_cached(next_ref)
        if isinstance(obj, dict) and (obj.get("status") or "").strip().lower() != "deleted":
            return obj

    # Fallback: any non-deleted task already occupying next link number in this chain.
    chain_id = (parent_task.get("chainID") or parent_task.get("chainid") or "").strip()
    if not chain_id:
        return None
    try:
        rows = _get_chain_export(chain_id, extra=f"link:{int(next_no)} status.not:deleted")
    except Exception:
        rows = []
    if not rows:
        return None

    for st in ("pending", "waiting", "completed"):
        for row in rows:
            if (row.get("status") or "").strip().lower() == st:
                return row
    return rows[0]


def _build_chain_indexes(chain: list[dict]) -> tuple[dict[int, list[dict]], dict[str, dict]]:
    """Build link-index and short-uuid index for quick in-memory lookups."""
    by_link: dict[int, list[dict]] = {}
    by_short: dict[str, dict] = {}
    for t in chain:
        ln = core.coerce_int(t.get("link"), None)
        if ln is not None:
            by_link.setdefault(ln, []).append(t)
        u = t.get("uuid")
        if isinstance(u, str) and u:
            by_short[u[:8]] = t
    return by_link, by_short


def _set_chain_cache(chain_id: str, chain: list[dict]) -> None:
    """Set per-run chain cache to avoid repeated task exports."""
    global _CHAIN_CACHE_CHAIN_ID, _CHAIN_CACHE, _CHAIN_BY_SHORT, _CHAIN_BY_UUID
    chain_copy = list(chain or [])
    _, by_short = _build_chain_indexes(chain_copy)
    by_uuid = {
        t.get("uuid"): t for t in chain_copy if isinstance(t.get("uuid"), str) and t.get("uuid")
    }
    with _CHAIN_CACHE_LOCK:
        _CHAIN_CACHE_CHAIN_ID = chain_id or ""
        _CHAIN_CACHE = chain_copy
        _CHAIN_BY_SHORT = by_short
        _CHAIN_BY_UUID = by_uuid
    _diag_count("chain_cache_seeded")


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
        out_list: list[tuple[int, int]] = []
        for it in v:
            out_list.extend(_norm_hhmm_list(it))
        return out_list
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

def _extract_time_slots_for_date(dnf, target_date, default_seed_date) -> list[tuple[int, int]]:
    """Extract time slots for terms that match target_date."""
    out: set[tuple[int, int]] = set()
    matched = False
    try:
        for term in dnf:
            if all(core.atom_matches_on(atom, target_date, default_seed_date) for atom in term):
                matched = True
                for atom in term:
                    mods = atom.get("mods") or {}
                    for hhmm in _norm_hhmm_list(mods.get("t")):
                        out.add(hhmm)
    except Exception:
        return []
    if matched:
        return sorted(out)
    return _extract_time_slots_from_dnf(dnf)

def _skip_reference_dt_local(
    dnf,
    end_local: "datetime",
    due_local: Optional["datetime"],
    default_seed_date,
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

    slots = _extract_time_slots_for_date(dnf, due_local.date(), default_seed_date)
    if len(slots) <= 1:
        return due_local

    if end_local.date() != due_local.date():
        return due_local

    end_hhmm = (end_local.hour, end_local.minute)
    prev_slots = [s for s in slots if s <= end_hhmm]
    if not prev_slots:
        return due_local

    hh, mm = prev_slots[-1]
    tz = end_local.tzinfo or _nautical_local_tz()
    return datetime.combine(end_local.date(), time(hh, mm), tzinfo=tz)

def _as_local_dt(d: datetime | None) -> datetime | None:
    if d is None:
        return None
    if d.tzinfo is None:
        return d.replace(tzinfo=timezone.utc).astimezone(_nautical_local_tz())
    return d.astimezone(_nautical_local_tz())


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
    tz = after_local_dt.tzinfo or _nautical_local_tz()
    slots = _extract_time_slots_from_dnf(dnf)

    # same-day: only if the expression hits on that date
    adate = after_local_dt.date()
    try:
        prev = adate - timedelta(days=1)
        nxt_date, _ = core.next_after_expr(dnf, prev, default_seed_date, seed_base=seed_base)
    except Exception:
        nxt_date = None

    if nxt_date == adate:
        slots = _extract_time_slots_for_date(dnf, adate, default_seed_date)
        if not slots:
            slots = [fallback_hhmm] if fallback_hhmm else [(0, 0)]
        for hh, mm in slots:
            cand = datetime.combine(adate, time(hh, mm), tzinfo=tz)
            if cand > after_local_dt:
                return cand

    # otherwise, find the next matching date strictly after adate
    nxt_date, _ = core.next_after_expr(dnf, adate, default_seed_date, seed_base=seed_base)
    slots = _extract_time_slots_for_date(dnf, nxt_date, default_seed_date)
    if not slots:
        slots = [fallback_hhmm] if fallback_hhmm else [(0, 0)]
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
        _diag(f"datetime parse value error: {e}")
        return (None, "DateTime parsing error")
    except TypeError as e:
        _diag(f"datetime parse type error: {e}")
        return (None, "DateTime type error")
    except Exception as e:
        _diag(f"datetime parse unexpected error: {e}")
        return (None, "Unexpected error parsing datetime")


def _validate_anchor_mode(mode_str: str) -> tuple[str, str | None]:
    """
    Validate and normalize anchor_mode. Returns (normalized_mode, error_msg).
    """
    raw = (mode_str or "").strip()
    if not raw:
        return ("", None)
    mode = raw.lower()
    if mode not in ("skip", "all", "flex"):
        return (
            "skip",
            f"anchor_mode must be 'skip', 'all', or 'flex' (got '{raw}'). Defaulting to 'skip'.",
        )
    return (mode, None)


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
        _diag(f"duration parse value error: {e}")
        return (None, "Duration parsing error")
    except TypeError as e:
        _diag(f"duration parse type error: {e}")
        return (None, "Duration type error")
    except Exception as e:
        _diag(f"duration parse unexpected error: {e}")
        return (None, "Unexpected error parsing duration")


def _anchor_mode_from_parent(parent: dict) -> str:
    mode = (parent.get("anchor_mode") or "skip").strip().lower()
    if mode not in ("skip", "all", "flex"):
        raise ValueError(f"anchor_mode must be 'skip', 'all', or 'flex', got '{mode}'")
    return mode


def _anchor_dnf_from_parent(parent: dict) -> tuple[str, list[list[dict]] | None]:
    expr_str = (parent.get("anchor") or "").strip()
    if not expr_str:
        return "", None
    try:
        return expr_str, _validate_anchor_expr_cached(expr_str)
    except Exception as e:
        raise ValueError(f"Invalid anchor expression '{expr_str}': {str(e)}")


def _anchor_parent_local_times(parent: dict):
    end_dt_utc, err = _safe_parse_datetime(parent.get("end"))
    if err:
        raise ValueError(f"end field: {err}")
    if not end_dt_utc:
        return None, None, None

    due_dt_utc, err = _safe_parse_datetime(parent.get("due"))
    if err:
        raise ValueError(f"due field: {err}")

    end_local = _tolocal(end_dt_utc)
    due_local = _tolocal(due_dt_utc) if due_dt_utc else end_local
    return end_local, due_local, due_dt_utc


def _anchor_due_mode_all(
    *,
    dnf,
    due_local,
    end_local,
    due_dt_utc,
    default_seed,
    seed_base,
    fallback_hhmm,
) -> tuple[object, dict]:
    info: dict[str, object] = {"mode": "all", "basis": None, "missed_count": 0, "missed_preview": []}
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
        info.update(
            basis="missed",
            missed_count=len(missed_dts),
            missed_preview=[x.isoformat() for x in missed_dts[:5]],
        )
        return missed_dts[0], info
    ref_local = _skip_reference_dt_local(
        dnf,
        end_local=end_local,
        due_local=(due_local if due_dt_utc else None),
        default_seed_date=default_seed,
    )
    nxt_local = _next_occurrence_after_local_dt(
        dnf,
        after_local_dt=ref_local,
        default_seed_date=default_seed,
        seed_base=seed_base,
        fallback_hhmm=fallback_hhmm,
    )
    info["basis"] = "after_due"
    return nxt_local, info


def _anchor_due_mode_flex(
    *,
    dnf,
    due_local,
    end_local,
    due_dt_utc,
    default_seed,
    seed_base,
    fallback_hhmm,
) -> tuple[object, dict]:
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
    info = {
        "mode": "flex",
        "basis": "flex",
        "missed_count": len(missed_dts),
        "missed_preview": [x.isoformat() for x in missed_dts[:5]],
    }
    return nxt_local, info


def _anchor_due_mode_skip(
    *,
    dnf,
    due_local,
    end_local,
    default_seed,
    seed_base,
    fallback_hhmm,
) -> tuple[object, dict]:
    nxt_local = _next_occurrence_after_local_dt(
        dnf,
        after_local_dt=(max(end_local, due_local) if due_local else end_local),
        default_seed_date=default_seed,
        seed_base=seed_base,
        fallback_hhmm=fallback_hhmm,
    )
    info = {"mode": "skip", "basis": "after_end", "missed_count": 0, "missed_preview": []}
    return nxt_local, info


def _anchor_due_for_mode(
    mode: str,
    *,
    dnf,
    due_local,
    end_local,
    due_dt_utc,
    default_seed,
    seed_base,
    fallback_hhmm,
) -> tuple[object, dict]:
    if mode == "all":
        return _anchor_due_mode_all(
            dnf=dnf,
            due_local=due_local,
            end_local=end_local,
            due_dt_utc=due_dt_utc,
            default_seed=default_seed,
            seed_base=seed_base,
            fallback_hhmm=fallback_hhmm,
        )
    if mode == "flex":
        return _anchor_due_mode_flex(
            dnf=dnf,
            due_local=due_local,
            end_local=end_local,
            due_dt_utc=due_dt_utc,
            default_seed=default_seed,
            seed_base=seed_base,
            fallback_hhmm=fallback_hhmm,
        )
    return _anchor_due_mode_skip(
        dnf=dnf,
        due_local=due_local,
        end_local=end_local,
        default_seed=default_seed,
        seed_base=seed_base,
        fallback_hhmm=fallback_hhmm,
    )


def _compute_anchor_child_due(parent: dict):
    """Return (next_due_utc, meta, dnf).

    The core recurrence engine computes *dates*; the hook expands into *datetimes* to
    respect multi-time lists: @t=HH:MM[,HH:MM...].
    """
    expr_str, dnf = _anchor_dnf_from_parent(parent)
    if not expr_str or not dnf:
        return (None, None, None)

    mode = _anchor_mode_from_parent(parent)
    end_local, due_local, due_dt_utc = _anchor_parent_local_times(parent)
    if not end_local:
        return (None, None, None)

    default_seed = due_local.date()
    seed_base = (parent.get("chainID") or "").strip() or "preview"
    fallback_hhmm = (due_local.hour, due_local.minute)
    nxt_local, info = _anchor_due_for_mode(
        mode,
        dnf=dnf,
        due_local=due_local,
        end_local=end_local,
        due_dt_utc=due_dt_utc,
        default_seed=default_seed,
        seed_base=seed_base,
        fallback_hhmm=fallback_hhmm,
    )

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

    seed_base = (task.get("chainID") or "").strip() or "preview"
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
_UDA_CARRY_SKIP_LOWER = {
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
    "nextlink",
    "prevlink",
    "link",
    "chain",
    "chainmax",
    "chainuntil",
    "chainid",
    "cp",
    "anchor",
    "anchor_mode",
    "due",
    "entry",
    "wait",
    "scheduled",
    "until",
}




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
        naive = dt_local_naive.replace(microsecond=0)
        aware0 = naive.replace(tzinfo=tz, fold=0)
        aware1 = naive.replace(tzinfo=tz, fold=1)

        # Ambiguous time (fall back): choose earlier instance (fold=0).
        if aware0.utcoffset() != aware1.utcoffset():
            return aware0.astimezone(timezone.utc)

        # Non-existent time (spring forward): shift forward to next valid minute.
        back = aware0.astimezone(timezone.utc).astimezone(tz)
        if back.replace(tzinfo=None) != naive:
            cand = naive
            for _ in range(180):
                cand += timedelta(minutes=1)
                aware = cand.replace(tzinfo=tz, fold=0)
                back = aware.astimezone(timezone.utc).astimezone(tz)
                if back.replace(tzinfo=None) == cand:
                    return aware.astimezone(timezone.utc)
            return aware0.astimezone(timezone.utc)

        return aware0.astimezone(timezone.utc)
    return dt_local_naive.replace(tzinfo=timezone.utc).replace(microsecond=0)


def _carry_relative_datetime(parent: dict, child: dict, child_due_utc: datetime, field: str) -> None:
    """Carry a datetime field forward relative to due, preserving local wall-clock offset.

    offset := (parent[field] - parent[due]) computed in local wall-clock space
    child[field] := child_due + offset (also in local wall-clock space)
    """
    if not isinstance(parent, dict) or not isinstance(child, dict):
        return
    if not _require_core():
        return
    if not (parent.get(field) and parent.get("due")):
        return

    p_due = core.parse_dt_any(parent.get("due"))
    p_val = core.parse_dt_any(parent.get(field))
    if not (p_due and p_val and isinstance(child_due_utc, datetime)):
        if _DEBUG_WAIT_SCHED:
            _set_wait_sched_debug(field, {
                "ok": False,
                "reason": "parse-failed",
                "parent_due": parent.get("due"),
                "parent_val": parent.get(field),
                "child_due": child.get("due") or (core.fmt_isoz(child_due_utc) if isinstance(child_due_utc, datetime) else None),
            })
        return

    try:
        # Do not carry absolute timestamps forward.
        child.pop(field, None)
        delta = _utc_to_local_naive(p_val) - _utc_to_local_naive(p_due)
        c_due_local = _utc_to_local_naive(child_due_utc)
        c_val_local = c_due_local + delta
        c_val_utc = _local_naive_to_utc(c_val_local)
        child[field] = core.fmt_isoz(c_val_utc)
        if _DEBUG_WAIT_SCHED:
            _set_wait_sched_debug(field, {
                "ok": True,
                "parent_due": parent.get("due"),
                "parent_val": parent.get(field),
                "delta": _fmt_td_dd_hhmm(delta),
                "child_due": child.get("due"),
                "child_val": child.get(field),
            })
    except Exception:
        # If anything goes wrong, do not mutate the child's field (leave inherited value).
        return


def _configured_recurrence_uda_fields(parent: dict) -> tuple[str, ...]:
    if not isinstance(parent, dict):
        return ()
    cfg = _RECURRENCE_UPDATE_UDAS if isinstance(_RECURRENCE_UPDATE_UDAS, (tuple, list)) else ()
    if not cfg:
        return ()
    parent_keys: dict[str, str] = {}
    for k in parent.keys():
        if isinstance(k, str) and k:
            parent_keys.setdefault(k.lower(), k)
    out: list[str] = []
    seen: set[str] = set()
    for name in cfg:
        lk = str(name or "").strip().lower()
        if not lk or lk in seen or lk in _UDA_CARRY_SKIP_LOWER:
            continue
        seen.add(lk)
        actual = parent_keys.get(lk)
        if actual:
            out.append(actual)
    return tuple(out)


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


    # Carry wait/scheduled forward relative to due (local wall-clock delta).
    _carry_relative_datetime(parent, child, child_due_utc, "wait")
    _carry_relative_datetime(parent, child, child_due_utc, "scheduled")
    for uda_field in _configured_recurrence_uda_fields(parent):
        _carry_relative_datetime(parent, child, child_due_utc, uda_field)

    if cpmax:
        child["chainMax"] = int(cpmax)
    if until_dt:
        child["chainUntil"] = core.fmt_isoz(until_dt)

    # [CHAINID] Inherit parent chainID (fallback to parent's short uuid)
    try:
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
      - When debug_wait_sched=true, we stash a short debug payload for the feedback panel.
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
            _set_wait_sched_debug(field, {
                "ok": False,
                "reason": "parse-failed",
                "parent_due": parent.get("due"),
                "parent_val": parent.get(field),
                "child_due": child.get("due") or (core.fmt_isoz(child_due_utc) if isinstance(child_due_utc, datetime) else None),
            })
        return

    delta = (p_val - p_due)
    c_val = (child_due_utc + delta).replace(microsecond=0)
    child[field] = core.fmt_isoz(c_val)

    if _DEBUG_WAIT_SCHED:
        delta_s = _fmt_td_dd_hhmm(delta)

        _set_wait_sched_debug(field, {
            "ok": True,
            "parent_due": parent.get("due"),
            "parent_val": parent.get(field),
            "delta": delta_s,
            "child_due": child.get("due"),
            "child_val": child.get(field),
        })

# ------------------------------------------------------------------------------
# End-of-chain summary + stats
# ------------------------------------------------------------------------------
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


def _fmt_td_compact_abs(delta: timedelta) -> str:
    s = _fmt_td_dd_hhmm(delta)
    return s[1:] if s and s[0] in "+-" else s


def _sort_chain_for_analytics(chain: list[dict]) -> list[dict]:
    def _link_sort_key(obj):
        ln = core.coerce_int(obj.get("link"), None)
        if ln is not None:
            return (0, ln)
        due = _dtparse(obj.get("due")) or datetime.max.replace(tzinfo=timezone.utc)
        return (1, due)
    try:
        return sorted(chain, key=_link_sort_key)
    except Exception:
        return chain[:]


def _chain_health_streak(completed_with_dates: list[dict], tol_secs: int) -> int:
    streak = 0
    for t in reversed(completed_with_dates):
        due = _dtparse(t.get("due"))
        end = _dtparse(t.get("end"))
        if not (due and end):
            continue
        diff = abs((end - due).total_seconds())
        if diff <= tol_secs:
            streak += 1
        else:
            break
    return streak


def _chain_health_completed_metrics(ordered: list[dict], tol_secs: int) -> dict:
    completed = [
        t for t in ordered
        if (t.get("status") or "").strip().lower() == "completed"
    ]

    completed_with_dates = []
    deltas = []
    for t in completed:
        due = _dtparse(t.get("due"))
        end = _dtparse(t.get("end"))
        if due and end:
            completed_with_dates.append(t)
            deltas.append((end - due).total_seconds())

    on_time_rate = None
    streak = 0
    vol = None
    if deltas:
        stats = _lateness_stats(completed_with_dates, tol_secs=tol_secs)
        on_time_rate = stats["on_time"] / max(1, stats["count"])
        if len(deltas) >= 2:
            try:
                import statistics
                vol = statistics.pstdev(deltas)
            except Exception:
                vol = None
        streak = _chain_health_streak(completed_with_dates, tol_secs)

    return {
        "on_time_rate": on_time_rate,
        "streak": streak,
        "vol": vol,
    }


def _chain_health_drift(ordered: list[dict], kind: str, task: dict) -> tuple[float | None, float | None]:
    drift_secs = None
    median_gap = None
    due_list = []
    for t in ordered:
        due = _dtparse(t.get("due"))
        if due:
            due_list.append(due)

    if len(due_list) < 2:
        return drift_secs, median_gap

    gaps = [
        (due_list[i] - due_list[i - 1]).total_seconds()
        for i in range(1, len(due_list))
        if due_list[i] and due_list[i - 1]
    ]
    gaps = [g for g in gaps if g > 0]
    if not gaps:
        return drift_secs, median_gap

    median_gap = _median(gaps)
    if kind == "cp":
        td = core.parse_cp_duration(task.get("cp") or "")
        if td:
            drift_secs = median_gap - td.total_seconds()
    elif len(gaps) >= 2:
        drift_secs = gaps[-1] - median_gap
    return drift_secs, median_gap


def _chain_health_clinical_text(
    on_time_rate: float | None,
    drift_secs: float | None,
    streak: int,
    vol: float | None,
) -> str | None:
    parts = []
    if on_time_rate is not None:
        parts.append(f"OT {int(round(100.0 * on_time_rate))}%")
    if drift_secs is not None:
        parts.append(f"Drift {_fmt_td_dd_hhmm(timedelta(seconds=drift_secs))}")
    if streak:
        parts.append(f"Streak {streak}")
    if isinstance(vol, (int, float)):
        parts.append(f"Vol {_fmt_td_compact_abs(timedelta(seconds=abs(vol)))}")
    return " | ".join(parts) if parts else None


def _chain_health_coach_text(
    kind: str,
    task: dict,
    on_time_rate: float | None,
    drift_secs: float | None,
    median_gap: float | None,
    streak: int,
    vol: float | None,
) -> str | None:
    issues = []
    tips = []
    positives = []

    if on_time_rate is not None:
        if on_time_rate < 0.6:
            issues.append("on-time rate is low")
            tips.append("try smaller scopes or later due times")
        elif on_time_rate < 0.8:
            issues.append("on-time is inconsistent")
            tips.append("adding a small buffer could help")
        else:
            positives.append("on-time is steady")

    if drift_secs is not None:
        base = None
        if kind == "cp":
            td = core.parse_cp_duration(task.get("cp") or "")
            base = td.total_seconds() if td else None
        else:
            base = median_gap
        if base:
            drift_warn = max(0.35 * base, 6 * 60 * 60)
            if abs(drift_secs) > drift_warn:
                issues.append("cadence is drifting")
                tips.append("review cp/anchors for a better fit")
            else:
                positives.append("cadence is stable")

    if isinstance(vol, (int, float)):
        if vol > 24 * 60 * 60:
            issues.append("timing is noisy")
            tips.append("add buffer or split tasks")
        elif vol < 6 * 60 * 60:
            positives.append("timing is consistent")

    if not issues:
        if streak >= 3:
            return (
                f"Chain looks healthy with a {streak}-link on-time streak; "
                "keep the current cadence."
            )
        if positives:
            return "Chain looks healthy; keep the current cadence."
        return None

    issue_txt = ", ".join(issues)
    tip_txt = "; ".join(tips[:2]) if tips else "keep an eye on due time fit"
    if streak >= 3:
        return (
            f"Chain needs attention ({issue_txt}); {tip_txt}, and keep the "
            f"{streak}-link on-time streak going."
        )
    return f"Chain needs attention ({issue_txt}); {tip_txt}."


def _chain_health_advice(
    chain: list[dict],
    kind: str,
    task: dict,
    tol_secs: int = _ANALYTICS_ONTIME_TOL_SECS,
    style: str = _ANALYTICS_STYLE,
) -> str | None:
    if not chain:
        return None

    ordered = _sort_chain_for_analytics(chain)
    metrics = _chain_health_completed_metrics(ordered, tol_secs)
    on_time_rate = metrics["on_time_rate"]
    streak = metrics["streak"]
    vol = metrics["vol"]
    drift_secs, median_gap = _chain_health_drift(ordered, kind, task)

    style = (style or "coach").strip().lower()
    if style == "clinical":
        return _chain_health_clinical_text(on_time_rate, drift_secs, streak, vol)
    return _chain_health_coach_text(
        kind,
        task,
        on_time_rate,
        drift_secs,
        median_gap,
        streak,
        vol,
    )


def _chain_integrity_collect(
    chain: list[dict],
    expected_chain_id: str | None,
) -> tuple[dict[str, dict], dict[int, dict], list[str], list[str]]:
    warnings: list[str] = []
    short_map: dict[str, dict] = {}
    link_map: dict[int, dict] = {}
    missing_link: list[str] = []
    for t in chain:
        if not isinstance(t, dict):
            continue
        uid = t.get("uuid")
        if uid:
            short_map[_short(uid)] = t

        link = core.coerce_int(t.get("link"), None)
        if link:
            if link in link_map:
                warnings.append(
                    f"duplicate link #{link} ({_short(link_map[link].get('uuid'))} vs {_short(uid)})"
                )
            else:
                link_map[link] = t
        elif uid:
            missing_link.append(_short(uid))

        if expected_chain_id is not None:
            cid = (t.get("chainID") or t.get("chainid") or "").strip()
            if not cid:
                warnings.append(f"missing chainID on {_short(uid)}")
            elif cid != expected_chain_id:
                warnings.append(f"chainID mismatch on {_short(uid)}")
    return short_map, link_map, missing_link, warnings


def _chain_integrity_missing_link_warning(missing_link: list[str]) -> list[str]:
    if not missing_link:
        return []
    sample = ", ".join(missing_link[:3])
    tail = "…" if len(missing_link) > 3 else ""
    return [f"missing link number on {sample}{tail}"]


def _chain_integrity_link_sequence_warnings(link_map: dict[int, dict]) -> list[str]:
    if not link_map:
        return []
    warnings: list[str] = []
    links_sorted = sorted(link_map.keys())
    if links_sorted[0] != 1:
        warnings.append(f"chain starts at link #{links_sorted[0]} (expected #1)")
    expected = set(range(links_sorted[0], links_sorted[-1] + 1))
    gaps = sorted(expected - set(links_sorted))
    if gaps:
        gap_list = ", ".join(str(g) for g in gaps[:5])
        tail = "…" if len(gaps) > 5 else ""
        warnings.append(f"missing link(s): {gap_list}{tail}")
    return warnings


def _chain_integrity_reciprocal_warnings(chain: list[dict], short_map: dict[str, dict]) -> list[str]:
    warnings: list[str] = []
    for t in chain:
        if not isinstance(t, dict):
            continue
        cur_short = _short(t.get("uuid"))
        prev_link = (t.get("prevLink") or "").strip()
        if prev_link:
            prev_task = short_map.get(prev_link)
            if not prev_task:
                warnings.append(f"{cur_short} prevLink {prev_link} not found")
            elif (prev_task.get("nextLink") or "").strip() != cur_short:
                warnings.append(f"{cur_short} prevLink {prev_link} not reciprocal")

        next_link = (t.get("nextLink") or "").strip()
        if next_link:
            next_task = short_map.get(next_link)
            if not next_task:
                warnings.append(f"{cur_short} nextLink {next_link} not found")
            elif (next_task.get("prevLink") or "").strip() != cur_short:
                warnings.append(f"{cur_short} nextLink {next_link} not reciprocal")
    return warnings


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    deduped: list[str] = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _chain_integrity_warnings(chain: list[dict], expected_chain_id: str | None = None) -> list[str]:
    if core is None:
        try:
            _load_core()
        except Exception:
            return []
    if not isinstance(chain, list) or not chain:
        return []

    short_map, link_map, missing_link, warnings = _chain_integrity_collect(chain, expected_chain_id)
    warnings.extend(_chain_integrity_missing_link_warning(missing_link))
    warnings.extend(_chain_integrity_link_sequence_warnings(link_map))
    warnings.extend(_chain_integrity_reciprocal_warnings(chain, short_map))
    return _dedupe_preserve_order(warnings)


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
            marker = "✓"
            line = f"{lab} {marker:<2} {end_s} {delta} [dim]{short}[/]"
            # Highlight the most recent task
            if no == get_link(chain_with_links[0]):
                line = f"[green]{line}[/]"
            top_lines.append(line)
        
        # Add ellipsis
        ellipsis_line = f"[dim]{' ' * (label_width + 4)}... ({len(chain_with_links) - 6} more tasks) ...[/dim]"
        
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
            marker = "✓"
            line = f"{lab} {marker:<2} {end_s} {delta} [dim]{short}[/]"
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
        marker = "✓"
        line = f"{lab} {marker:<2} {end_s} {delta} [dim]{short}[/]"
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


def _end_summary_current(current: dict, current_task: dict | None) -> dict:
    return current_task if current_task else current


def _end_summary_chain_id_row(actual_current: dict) -> str:
    return (actual_current.get("chainID") or actual_current.get("chainid") or "").strip()


def _end_summary_sorted_chain(chain_id: str, actual_current: dict) -> list[dict]:
    chain = tw_export_chain_required(actual_current)
    if actual_current and chain:
        for i, task in enumerate(chain):
            if task.get("uuid") == actual_current.get("uuid"):
                chain[i] = actual_current
                break
    try:
        chain = _sort_chain_for_analytics(chain)
    except Exception:
        pass
    return chain


def _end_summary_span_fields(chain_id: str, chain: list[dict]) -> tuple[datetime | None, datetime | None, str]:
    first_task = _export_chain_endpoint(chain_id, "first")
    last_task = _export_chain_endpoint(chain_id, "last")
    first = _dtparse((first_task or {}).get("due")) if first_task else (_dtparse(chain[0].get("due")) if chain else None)
    last = _dtparse((last_task or {}).get("end")) if last_task else (_dtparse(chain[-1].get("end")) if chain else None)
    span = "–"
    if first and last:
        span = (
            _human_delta(first, last, prefer_months=True)
            .replace("in ", "")
            .replace("overdue by ", "")
        )
    return first, last, span


def _end_summary_kind_rows(rows: list[tuple[str, str]], kind: str, current: dict) -> None:
    if kind == "anchor":
        expr = (current.get("anchor") or "").strip()
        mode = (current.get("anchor_mode") or "skip").lower()
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
        return
    rows.append(("Period", current.get("cp") or "–"))


def _end_summary_stats_rows(rows: list[tuple[str, str]], chain: list[dict], now_utc) -> None:
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


def _end_summary_limits_row(rows: list[tuple[str, str]], current: dict) -> None:
    cpmax = core.coerce_int(current.get("chainMax"), 0)
    until = _dtparse(current.get("chainUntil"))
    lims = []
    if cpmax:
        lims.append(f"max {cpmax}")
    if until:
        lims.append(f"until {core.fmt_dt_local(until)}")
    rows.append(("Limits", " | ".join(lims) if lims else "–"))


def _end_chain_summary(current: dict, reason: str, now_utc, current_task: dict = None) -> None:
    actual_current = _end_summary_current(current, current_task)
    kind_anchor = bool((actual_current.get("anchor") or "").strip())
    kind = "anchor" if kind_anchor else "cp"

    chain_id = _end_summary_chain_id_row(actual_current)
    if not chain_id:
        _panel(
            "⚠ Chain summary skipped",
            [
                ("Reason", "ChainID is required in v3+ and legacy link-walk is removed."),
                ("Fix", "Run tools/nautical_backfill_chainid.py."),
            ],
            kind="warning",
        )
        return

    chain = _end_summary_sorted_chain(chain_id, actual_current)

    L = core.coerce_int(current.get("link"), len(chain))
    root = _short(_root_uuid_from(current))
    cur_s = _short(current.get("uuid"))
    first, last, span = _end_summary_span_fields(chain_id, chain)

    rows = []
    rows.append(("Reason", reason))
    rows.append(("Root", _format_root_and_age(current, now_utc)))

    chain_display = f"{root} … {cur_s}  [dim](#{L}, {len(chain)} tasks"
    if len(chain) >= _MAX_CHAIN_WALK:
        chain_display += f", truncated at {_MAX_CHAIN_WALK})"
    else:
        chain_display += ")"
    rows.append(("Chain", chain_display))

    _end_summary_kind_rows(rows, kind, current)

    if first:
        rows.append(("First due", core.fmt_dt_local(first)))
    if last:
        rows.append(("Last end", core.fmt_dt_local(last)))
    rows.append(("Span", span))

    _end_summary_stats_rows(rows, chain, now_utc)
    _end_summary_limits_row(rows, current)

    tail = _last_n_timeline(chain, n=6)
    if tail:
        rows.append(("History", "\n".join(tail)))

    rows = _format_chain_summary_rows(rows)
    _panel("⛔ Chain finished – summary", rows, kind="summary")



# ------------------------------------------------------------------------------
# Timeline (capped) — no dependency on core.next_anchor_after
# ------------------------------------------------------------------------------

def _timeline_styles(task: dict, kind: str) -> tuple[str, str, str, str]:
    if kind == "cp":
        prev_style = "dim green"
        cur_style = "spring_green1"
        next_style = "bold yellow"
    else:
        prev_style = "sky_blue3"
        cur_style = "spring_green1"
        next_style = "bold yellow"
    future_style = _future_style_for_chain(task, kind)
    return prev_style, cur_style, next_style, future_style


def _timeline_initial_items(task: dict, cur_no: int, nxt_no: int, child_due_utc, child_short: str):
    items = []
    prevs = _collect_prev_two(task)
    prev_count = len(prevs)
    for idx, obj in enumerate(prevs):
        no = core.coerce_int(obj.get("link"), None) or (cur_no - (prev_count - idx))
        end_dt = _dtparse(obj.get("end"))
        items.append((no, end_dt, obj, "prev"))
    cur_end = _dtparse(task.get("end"))
    items.append((cur_no, cur_end, task, "current"))
    items.append((nxt_no, child_due_utc, {"uuid": child_short}, "next"))
    return items


def _timeline_future_cp_items(task: dict, child_due_utc, *, start_no: int, allowed_future: int, cap_no: int | None):
    td = core.parse_cp_duration(task.get("cp") or "")
    if not td:
        return []
    items = []
    fut_dt = child_due_utc
    fut_no = start_no
    secs = int(td.total_seconds())
    iterations = 0
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
    return items


def _timeline_future_anchor_items(
    task: dict,
    dnf,
    child_due_utc,
    *,
    start_no: int,
    allowed_future: int,
    cap_no: int | None,
):
    items = []
    fut_no = start_no
    seed_base = (task.get("chainID") or "").strip() or "preview"
    nxt_local = _to_local_cached(child_due_utc)
    fallback_hhmm = (nxt_local.hour, nxt_local.minute)
    due0, _ = _safe_parse_datetime(task.get("due"))
    default_seed = _to_local_cached(due0 or child_due_utc).date()
    after_local = nxt_local
    iterations = 0
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
    return items


def _timeline_base_line(
    no: int,
    dt,
    obj: dict,
    item_type: str,
    *,
    task: dict,
    cap_no: int | None,
    prev_style: str,
    cur_style: str,
    next_style: str,
    future_style: str,
) -> str:
    if item_type == "prev":
        end_dt = _dtparse(obj.get("end"))
        due_dt = _dtparse(obj.get("due"))
        delta = _fmt_on_time_delta(due_dt, end_dt)
        end_s = _fmtlocal(end_dt) if end_dt else "(no end)"
        short = _short(obj.get("uuid"))
        return f"[{prev_style}]{no:>2} {'✓':<2}{end_s} {delta} {short}[/]"

    if item_type == "current":
        cur_end = _dtparse(task.get("end"))
        cur_due = _dtparse(task.get("due"))
        cur_delta = _fmt_on_time_delta(cur_due, cur_end)
        cur_end_s = _fmtlocal(cur_end) if cur_end else "(no end)"
        return f"[{cur_style}]{no:>2} {'✓':<2}{cur_end_s} {cur_delta} {_short(task.get('uuid'))}[/]"

    if item_type == "next":
        is_last = cap_no is not None and no == cap_no
        next_text = f"{no:>2} {'►':<2}{core.fmt_dt_local(dt)} {_short(obj.get('uuid'))}"
        if is_last:
            return f"[{next_style}]{next_text} [bold red](last link)[/][/]"
        return f"[{next_style}]{next_text}[/]"

    is_last = cap_no is not None and no == cap_no
    future_text = f"{no:>2} {'»':<2}{core.fmt_dt_local(dt)}"
    if is_last:
        return f"[{future_style}]{future_text} [bold red](last link)[/][/]"
    return f"[{future_style}]{future_text}[/]"


def _timeline_with_gap(
    base_line: str,
    *,
    idx: int,
    items: list[tuple[int, object, dict, str]],
    show_gaps: bool,
    kind: str,
    round_anchor_gaps: bool,
) -> str:
    if not show_gaps or idx >= len(items) - 1:
        return base_line
    dt = items[idx][1]
    next_dt = items[idx + 1][1]
    if not (dt and next_dt):
        return base_line
    gap_text = _format_gap(dt, next_dt, kind, round_anchor_gaps)
    if not gap_text:
        return base_line
    return f"{base_line}{gap_text}"


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
    if not _require_core():
        return []
    cur_no = core.coerce_int(task.get("link") if cur_no is None else cur_no, 1)
    nxt_no = cur_no + 1
    allowed_future = next_count if cap_no is None else max(0, min(next_count, cap_no - nxt_no))
    prev_style, cur_style, next_style, future_style = _timeline_styles(task, kind)
    items = _timeline_initial_items(task, cur_no, nxt_no, child_due_utc, child_short)
    if allowed_future > 0:
        if kind == "cp":
            items.extend(
                _timeline_future_cp_items(
                    task,
                    child_due_utc,
                    start_no=nxt_no,
                    allowed_future=allowed_future,
                    cap_no=cap_no,
                )
            )
        else:
            items.extend(
                _timeline_future_anchor_items(
                    task,
                    dnf,
                    child_due_utc,
                    start_no=nxt_no,
                    allowed_future=allowed_future,
                    cap_no=cap_no,
                )
            )

    lines = []
    for i, (no, dt, obj, item_type) in enumerate(items):
        base_line = _timeline_base_line(
            no,
            dt,
            obj,
            item_type,
            task=task,
            cap_no=cap_no,
            prev_style=prev_style,
            cur_style=cur_style,
            next_style=next_style,
            future_style=future_style,
        )
        lines.append(
            _timeline_with_gap(
                base_line,
                idx=i,
                items=items,
                show_gaps=show_gaps,
                kind=kind,
                round_anchor_gaps=round_anchor_gaps,
            )
        )
    return lines

def _got_anchor_invalid(msg: str) -> None:
    _fail_and_exit("Invalid anchor", msg)


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
    seed_base = (task.get("chainID") or "").strip() or "preview"

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

def _extra_safe(extra: str) -> bool:
    return _parse_extra_tokens(extra) is not None


def _parse_extra_tokens(extra: str | None) -> list[str] | None:
    """Parse extra Taskwarrior filters in strict token form: key:value."""
    hook_support = _load_hook_support()
    if hook_support is not None:
        return hook_support.parse_extra_tokens(extra)
    if extra is None:
        return []
    if not isinstance(extra, str):
        return None
    s = extra.strip()
    if not s:
        return []
    out: list[str] = []
    for tok in s.split():
        if tok.startswith("+"):
            tag = tok[1:]
            if not tag or re.fullmatch(r"[A-Za-z0-9_.-]+", tag) is None:
                return None
            out.append(tok)
            continue
        if tok.startswith("-"):
            return None
        if ":" not in tok:
            return None
        key, value = tok.split(":", 1)
        if not key or not value:
            return None
        if re.fullmatch(r"[A-Za-z0-9_.-]+", key) is None:
            return None
        if re.fullmatch(r"[A-Za-z0-9_.:@%+,-]+", value) is None:
            return None
        out.append(f"{key}:{value}")
    return out

def _chain_export_timeout(chain_id: str) -> float:
    global _CHAIN_EXPORT_TIMEOUT_FLOOR
    base = float(_CHAIN_EXPORT_TIMEOUT_BASE)
    per_100 = float(_CHAIN_EXPORT_TIMEOUT_PER_100)
    max_t = float(_CHAIN_EXPORT_TIMEOUT_MAX)
    est = base
    with _CHAIN_CACHE_LOCK:
        cache_match = bool(chain_id and _CHAIN_CACHE_CHAIN_ID == chain_id and _CHAIN_CACHE)
        cache_len = len(_CHAIN_CACHE) if cache_match else 0
    if cache_match:
        extra = max(0, cache_len // 100)
        est = base + (extra * per_100)
    adaptive = 0.0
    if _CHAIN_EXPORT_TIMES:
        try:
            times = sorted(t for t in _CHAIN_EXPORT_TIMES if t > 0)
        except Exception:
            times = []
        if times:
            idx = int(0.95 * (len(times) - 1))
            p95 = times[max(0, min(idx, len(times) - 1))]
            adaptive = p95 * 2.0
    floor = _CHAIN_EXPORT_TIMEOUT_FLOOR
    if floor < base:
        floor = base
        _CHAIN_EXPORT_TIMEOUT_FLOOR = base
    if est < base:
        est = base
    timeout = max(est, adaptive, floor)
    if timeout > max_t:
        timeout = max_t
    return timeout

def _tw_export_chain_args(
    chain_id: str,
    *,
    since: datetime | None,
    extra: str | None,
    limit: int | None,
) -> list[str] | None:
    hook_support = _load_hook_support()
    if hook_support is not None:
        return hook_support.build_chain_export_args(
            task_cmd_prefix=_task_cmd_prefix(),
            chain_id=chain_id,
            since=since,
            extra=extra,
            limit=limit,
            parse_extra_tokens=_parse_extra_tokens,
            diag=_diag,
        )
    args = _task_cmd_prefix() + ["rc.hooks=off", "rc.json.array=on", "rc.verbose=nothing", f"chainID:{chain_id}"]
    if since:
        args.append(f"modified.after:{since.strftime('%Y-%m-%dT%H:%M:%S')}")
    if limit and isinstance(limit, int) and limit > 0:
        args.append(f"limit:{limit}")
    if extra:
        extra_tokens = _parse_extra_tokens(extra)
        if extra_tokens is None:
            _diag(f"tw_export_chain rejected extra: {extra!r}")
            return None
        args += extra_tokens
    args.append("export")
    return args


def _tw_export_chain_success(elapsed: float) -> None:
    global _CHAIN_EXPORT_TIMEOUT_FLOOR
    if elapsed > 0:
        _CHAIN_EXPORT_TIMES.append(elapsed)
        if len(_CHAIN_EXPORT_TIMES) > _CHAIN_EXPORT_TIMES_MAX:
            del _CHAIN_EXPORT_TIMES[:len(_CHAIN_EXPORT_TIMES) - _CHAIN_EXPORT_TIMES_MAX]
    if _CHAIN_EXPORT_TIMEOUT_FLOOR > _CHAIN_EXPORT_TIMEOUT_BASE:
        _CHAIN_EXPORT_TIMEOUT_FLOOR = max(_CHAIN_EXPORT_TIMEOUT_BASE, _CHAIN_EXPORT_TIMEOUT_FLOOR * 0.9)


def _tw_export_chain_failure(chain_id: str, err: str, timeout: float) -> None:
    global _CHAIN_EXPORT_TIMEOUT_FLOOR
    if "timeout" in (err or "").lower():
        _CHAIN_EXPORT_TIMEOUT_FLOOR = min(
            _CHAIN_EXPORT_TIMEOUT_MAX,
            max(_CHAIN_EXPORT_TIMEOUT_FLOOR, timeout * 1.5),
        )
    _diag(f"tw_export_chain failed (chainID={chain_id}): {err.strip()}")
    if chain_id and chain_id in _WARNED_CHAIN_EXPORT:
        return
    if chain_id:
        _WARNED_CHAIN_EXPORT.add(chain_id)
    if _is_lock_error(err):
        reason = "Taskwarrior lock active"
    else:
        reason = (err or "").strip() or "task export failed"
    _panel("⚠ Chain export failed", [("ChainID", chain_id), ("Reason", reason)], kind="warning")


def _tw_export_chain_parse(out: str) -> list[dict]:
    hook_support = _load_hook_support()
    if hook_support is not None:
        return hook_support.parse_export_array(out, diag=_diag)
    try:
        data = json.loads(out.strip() or "[]")
        return data if isinstance(data, list) else [data]
    except Exception as e:
        _diag(f"tw_export_chain JSON parse failed: {e}")
        return []


def tw_export_chain(chain_id: str, since: datetime | None = None, extra: str | None = None, env=None, limit: int | None = None) -> list[dict]:
    if not chain_id:
        return []
    args = _tw_export_chain_args(chain_id, since=since, extra=extra, limit=limit)
    if args is None:
        return []

    start = _time.perf_counter()
    timeout = _chain_export_timeout(chain_id)
    ok, out, err = _run_task(
        args,
        env=env,
        timeout=timeout,
        retries=1,
        use_tempfiles=True,
    )
    elapsed = _time.perf_counter() - start
    if ok:
        _tw_export_chain_success(elapsed)
        return _tw_export_chain_parse(out)
    _tw_export_chain_failure(chain_id, err, timeout)
    return []


def _export_chain_endpoint(chain_id: str, direction: str) -> dict | None:
    """Return the first/last chain task using a minimal export."""
    modify_queries = _load_modify_queries()
    if modify_queries is not None:
        hook_support = _load_hook_support()
        parser = (hook_support.parse_export_array if hook_support is not None else _tw_export_chain_parse)
        return modify_queries.export_chain_endpoint(
            chain_id,
            direction,
            run_task=_run_task,
            task_cmd_prefix=_task_cmd_prefix(),
            parse_export_array=parser,
            diag=_diag,
            timeout=3.0,
            retries=1,
        )
    if not chain_id:
        return None
    sort_dir = "+" if direction == "first" else "-"
    args = _task_cmd_prefix() + [
        "rc.hooks=off",
        "rc.json.array=on",
        "rc.verbose=nothing",
        f"chainID:{chain_id}",
        f"sort:link{sort_dir}",
        "limit:1",
        "export",
    ]
    ok, out, err = _run_task(args, env=None, timeout=3.0, retries=1)
    if not ok:
        _diag(f"chain endpoint export failed ({direction}): {err.strip()}")
        return None
    try:
        data = json.loads(out.strip() or "[]")
        if isinstance(data, list) and data:
            return data[0]
    except Exception:
        pass
    return None

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def _is_non_completion_modify(old: dict, new: dict) -> bool:
    return (old.get("status") == new.get("status")) or (new.get("status") != "completed")


def _stamp_chain_id_if_new_nautical(old: dict, new: dict) -> None:
    # [CHAINID] stamp only when task just became nautical and has no chainID/links
    try:
        became_anchor = (not (old.get("anchor") or "").strip()) and ((new.get("anchor") or "").strip())
        became_cp = (not (old.get("cp") or "").strip()) and ((new.get("cp") or "").strip())
        already_chain = bool((new.get("chainID") or "").strip())
        linked_already = bool((new.get("prevLink") or new.get("nextLink") or "").strip())
        if (became_anchor or became_cp) and not already_chain and not linked_already:
            new["chainID"] = core.short_uuid(new.get("uuid"))
    except Exception:
        pass


def _resolve_child_id(child_short: str, deferred_spawn: bool, chain_by_short: dict | None) -> str:
    child_id = ""
    if not deferred_spawn and chain_by_short:
        child_id = str(chain_by_short.get(child_short, {}).get("id", "") or "")
    if not deferred_spawn and not child_id:
        child_obj = _export_uuid_short_cached(child_short)
        child_id = child_obj.get("id", "") if child_obj else ""
    return child_id


def _anchor_mode_tag(new: dict) -> str:
    return {
        "skip": "[cyan]SKIP[/]",
        "all": "[yellow]ALL[/]",
        "flex": "[magenta]FLEX[/]",
    }.get((new.get("anchor_mode") or "skip").lower(), "[cyan]SKIP[/]")


def _append_wait_sched_feedback_rows(fb: list[tuple[str, object]]) -> None:
    if not (_DEBUG_WAIT_SCHED and _LAST_WAIT_SCHED_DEBUG):
        return
    for _fld in ("scheduled", "wait"):
        d = _LAST_WAIT_SCHED_DEBUG.get(_fld)
        if not d:
            continue
        if d.get("ok"):
            fb.append(
                (
                    f"{_fld} carry",
                    f"Δ {d.get('delta')}  parent {d.get('parent_val')} vs {d.get('parent_due')}  →  child {d.get('child_val')}",
                )
            )
        else:
            fb.append(
                (
                    f"{_fld} carry",
                    f"[yellow]skip[/] ({d.get('reason')})  parent {d.get('parent_val')} vs {d.get('parent_due')}",
                )
            )


def _append_sanitised_fields_row(fb: list[tuple[str, object]], stripped_attrs: list[str]) -> None:
    if stripped_attrs:
        fb.append(
            (
                "Sanitised",
                f"Removed unknown fields: {', '.join(sorted(stripped_attrs))}",
            )
        )


def _append_integrity_warnings_row(
    fb: list[tuple[str, object]],
    integrity_warnings: list[str] | None,
) -> None:
    if not integrity_warnings:
        return
    warn_list = integrity_warnings[:4]
    if len(integrity_warnings) > 4:
        warn_list.append(f"...and {len(integrity_warnings) - 4} more")
    fb.append(("Integrity", "\n".join(warn_list)))


def _append_link_status_rows(fb: list[tuple[str, object]], cap_no: int | None, base_no: int) -> None:
    if not cap_no:
        return
    if base_no >= cap_no:
        fb.append(("Link status", "[bold red]This was the last link[/]"))
    elif base_no == cap_no - 1:
        fb.append(("Link status", "[yellow]This was the second-to-last link[/]"))
    fb.append(("Links left", f"{max(0, cap_no - base_no)} left (cap #{cap_no})"))


def _append_final_rows(
    fb: list[tuple[str, object]],
    finals: list[tuple[str, object]],
    now_utc,
) -> None:
    for label, when in finals:
        fb.append(
            (
                f"Final ({label})",
                f"{core.fmt_dt_local(when)}  ({_human_delta(now_utc, when, True)})",
            )
        )


def _append_anchor_intent_row(
    fb: list[tuple[str, object]],
    *,
    child_short: str,
    deferred_spawn: bool,
    chain_by_short: dict | None,
    spawn_intent_id: str | None,
) -> None:
    _ = _resolve_child_id(child_short, deferred_spawn, chain_by_short)
    if deferred_spawn and os.environ.get("NAUTICAL_DIAG") == "1" and spawn_intent_id:
        fb.append(("Intent", spawn_intent_id))


def _append_anchor_timeline_rows(
    fb: list[tuple[str, object]],
    *,
    new: dict,
    child_due,
    child_short: str,
    dnf,
    cap_no: int | None,
    base_no: int,
    expr_str: str,
) -> None:
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


def _emit_anchor_completion_panel(
    *,
    title: str,
    fb: list[tuple[str, object]],
    base_no: int,
    new: dict,
    child_due,
    child_short: str,
    now_utc,
    cap_no: int | None,
    until_dt,
    until_cap_no: int | None,
) -> None:
    if (core.PANEL_MODE or "").strip().lower() == "line":
        line = _format_line_preview(
            base_no,
            new,
            child_due,
            child_short,
            now_utc,
            cap_no=cap_no,
            until_dt=until_dt,
            until_no=until_cap_no,
        )
        _panel_line(title, line, kind="preview_anchor")
        return
    if _CHAIN_COLOR_PER_CHAIN:
        chain_colour = _chain_colour_for_task(new, "anchor")
        _panel(
            title,
            fb,
            kind="preview_anchor",
            border_style=chain_colour,
            title_style=chain_colour,
        )
        return
    _panel(title, fb, kind="preview_anchor")


def _render_anchor_completion_feedback(
    *,
    new: dict,
    child: dict,
    child_due,
    child_short: str,
    next_no: int,
    parent_short: str,
    cap_no: int | None,
    finals: list[tuple[str, object]],
    now_utc,
    until_dt,
    until_cap_no: int | None,
    dnf,
    meta: dict,
    stripped_attrs: list[str],
    deferred_spawn: bool,
    spawn_intent_id: str | None,
    chain_by_short: dict | None,
    analytics_advice: str | None,
    integrity_warnings: list[str] | None,
    base_no: int,
) -> None:
    fb = []
    anchor_raw = (new.get("anchor") or "").strip()
    expr_str = _strip_quotes(anchor_raw)
    mode_tag = _anchor_mode_tag(new)
    fb.append(("Pattern", f"{expr_str}  {mode_tag}"))
    fb.append(("Natural", core.describe_anchor_dnf(dnf, new)))
    fb.append(("Basis", _pretty_basis_anchor(meta, new)))
    fb.append(("Root", _format_root_and_age(new, now_utc)))

    _append_wait_sched_feedback_rows(fb)
    _append_sanitised_fields_row(fb, stripped_attrs)

    delta = core.humanize_delta(now_utc, child_due, use_months_days=core.expr_has_m_or_y(dnf))
    fb.append(("Next Due", f"{core.fmt_dt_local(child_due)}  ({delta})"))
    if analytics_advice:
        fb.append(("Analytics", analytics_advice))
    _append_integrity_warnings_row(fb, integrity_warnings)
    _append_next_wait_sched_rows(fb, child, child_due)

    _append_link_status_rows(fb, cap_no, base_no)
    _append_final_rows(fb, finals, now_utc)
    _append_anchor_intent_row(
        fb,
        child_short=child_short,
        deferred_spawn=deferred_spawn,
        chain_by_short=chain_by_short,
        spawn_intent_id=spawn_intent_id,
    )

    title = f"⚓︎ Next anchor  #{next_no}  {parent_short} → {child_short}"
    _append_anchor_timeline_rows(
        fb,
        new=new,
        child_due=child_due,
        child_short=child_short,
        dnf=dnf,
        cap_no=cap_no,
        base_no=base_no,
        expr_str=expr_str,
    )

    fb = _format_next_anchor_rows(fb)
    _emit_anchor_completion_panel(
        title=title,
        fb=fb,
        base_no=base_no,
        new=new,
        child_due=child_due,
        child_short=child_short,
        now_utc=now_utc,
        cap_no=cap_no,
        until_dt=until_dt,
        until_cap_no=until_cap_no,
    )


def _render_cp_completion_feedback(
    *,
    new: dict,
    child: dict,
    child_due,
    child_short: str,
    next_no: int,
    parent_short: str,
    cap_no: int | None,
    finals: list[tuple[str, object]],
    now_utc,
    until_dt,
    until_cap_no: int | None,
    meta: dict,
    deferred_spawn: bool,
    spawn_intent_id: str | None,
    chain_by_short: dict | None,
    analytics_advice: str | None,
    integrity_warnings: list[str] | None,
    base_no: int,
) -> None:
    fb = []
    delta = core.humanize_delta(now_utc, child_due, use_months_days=False)
    fb.append(("Period", new.get("cp")))
    fb.append(("Basis", _pretty_basis_cp(new, meta)))
    fb.append(("Root", _format_root_and_age(new, now_utc)))
    fb.append(("Next Due", f"{core.fmt_dt_local(child_due)}  ({delta})"))
    if analytics_advice:
        fb.append(("Analytics", analytics_advice))
    if integrity_warnings:
        warn_list = integrity_warnings[:4]
        if len(integrity_warnings) > 4:
            warn_list.append(f"...and {len(integrity_warnings) - 4} more")
        fb.append(("Integrity", "\n".join(warn_list)))
    _append_next_wait_sched_rows(fb, child, child_due)

    if cap_no:
        if base_no >= cap_no:
            fb.append(("Link status", "[bold red]This was the last link[/]"))
        elif base_no == cap_no - 1:
            fb.append(("Link status", "[yellow]Next link is the last in the chain.[/]"))
        fb.append(("Links left", f"{max(0, cap_no - base_no)} left (cap #{cap_no})"))
    else:
        fb.append(("Limits", "—"))

    for label, when in finals:
        fb.append(
            (
                f"Final ({label})",
                f"{core.fmt_dt_local(when)}  ({_human_delta(now_utc, when, True)})",
            )
        )

    child_id = _resolve_child_id(child_short, deferred_spawn, chain_by_short)
    if deferred_spawn and os.environ.get("NAUTICAL_DIAG") == "1" and spawn_intent_id:
        fb.append(("Intent", spawn_intent_id))

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

    if (core.PANEL_MODE or "").strip().lower() == "line":
        line = _format_line_preview(
            base_no,
            new,
            child_due,
            child_short,
            now_utc,
            cap_no=cap_no,
            until_dt=until_dt,
            until_no=until_cap_no,
        )
        _panel_line(title, line, kind="preview_cp")
    elif _CHAIN_COLOR_PER_CHAIN:
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


def _non_completion_anchor_error_message(anchor_expr: str, default_msg: str) -> str:
    has_type_colon = bool(
        re.search(r"(?:^|[^A-Za-z])(w|m|y)(?:/\d+)?:", anchor_expr, re.IGNORECASE)
    )
    if has_type_colon:
        return default_msg
    if re.match(r"^(mon|tue|wed|thu|fri|sat|sun)\b", anchor_expr, re.IGNORECASE):
        return (
            "Weekly anchors must start with 'w:'. "
            "Examples: 'w:mon..fri' or 'w:mon,tue,wed,thu,fri'."
        )
    return (
        "Anchors must start with 'w:', 'm:' or 'y:'. "
        "Examples: 'w:mon', 'm:-1', 'y:06-01'."
    )


def _non_completion_anchor_mode(old: dict, new: dict) -> str:
    anchor_mode_raw = (new.get("anchor_mode") or old.get("anchor_mode") or "").strip()
    mode_norm, warn_msg = _validate_anchor_mode(anchor_mode_raw)
    if warn_msg:
        _panel("⚠ Anchor mode", [("Warning", warn_msg)], kind="warning")
        new["anchor_mode"] = mode_norm
    elif (new.get("anchor_mode") or "").strip():
        new["anchor_mode"] = mode_norm
    return ((mode_norm or anchor_mode_raw or "").strip().upper() or "ALL")


def _non_completion_validate_anchor_cache(new: dict, old: dict, anchor_expr: str) -> None:
    _, warns = core.lint_anchor_expr(anchor_expr)
    if warns:
        _panel("ℹ️  Lint", [("Hint", w) for w in warns], kind="note")

    anchor_mode = _non_completion_anchor_mode(old, new)
    due_dt = _safe_dt(new.get("due") or old.get("due"))
    if core.ENABLE_ANCHOR_CACHE:
        _ = core.build_and_cache_hints(anchor_expr, anchor_mode, default_due_dt=due_dt)
    else:
        _ = core.validate_anchor_expr_strict(anchor_expr)


def _non_completion_validate_anchor(old: dict, new: dict, new_anchor: str) -> None:
    try:
        _non_completion_validate_anchor_cache(new, old, new_anchor)
    except TypeError:
        _ = core.validate_anchor_expr_strict(new_anchor)
    except Exception as e:
        _got_anchor_invalid(_non_completion_anchor_error_message(new_anchor, str(e)))


def _non_completion_reject_conflicting_types(new_anchor: str, new_cp: str) -> None:
    if new_anchor and new_cp:
        _fail_and_exit("Invalid chain config", "anchor and cp cannot both be set; clear one")


def _handle_non_completion_modify(old: dict, new: dict) -> None:
    anchor_raw = (new.get("anchor") or "").strip()
    new_anchor = _strip_quotes(anchor_raw)

    if new_anchor:
        _non_completion_validate_anchor(old, new, new_anchor)

    cp_raw = (new.get("cp") or "").strip()
    new_cp = _strip_quotes(cp_raw)
    _non_completion_reject_conflicting_types(new_anchor, new_cp)

    _stamp_chain_id_if_new_nautical(old, new)
    _print_task(new)


def _completion_validate_cp_and_anchor(old: dict, new: dict) -> tuple[str, str]:
    # If we reach here, the task is being completed
    # Now we should validate CP (in addition to anchor which was already validated on modify)
    cp_raw = (new.get("cp") or "").strip()
    new_cp = _strip_quotes(cp_raw)
    anchor_raw = (new.get("anchor") or "").strip()
    new_anchor = _strip_quotes(anchor_raw)
    if new_anchor and new_cp:
        _fail_and_exit("Invalid chain config", "anchor and cp cannot both be set; clear one")

    if new_cp:
        # Validate CP on completion
        try:
            td = core.parse_cp_duration(new_cp)
            if td is None:
                raise ValueError(f"Invalid duration format '{new_cp}'")
        except ValueError as e:
            _fail_and_exit("Invalid CP", str(e))
        except Exception as e:
            _diag(f"cp parse unexpected error: {e}")
            _fail_and_exit("CP parsing error", "Unexpected error while parsing cp")

        # Deep checks only if fields changed
        if _field_changed(old, new, "anchor") or _field_changed(old, new, "anchor_mode"):
            if new_anchor:
                _validate_anchor_on_modify(new_anchor)
            # _ensure_acf(new)  # keep in-memory ACF consistent (no UDA writes)

        if (
            _field_changed(old, new, "cp")
            or _field_changed(old, new, "chainMax")
            or _field_changed(old, new, "chainUntil")
        ) and new_cp:
            _validate_cp_on_modify(new_cp, new.get("chainMax"), new.get("chainUntil"))

        _stamp_chain_id_if_new_nautical(old, new)

    return new_cp, new_anchor


def _completion_link_numbers_or_fail(new: dict) -> tuple[int, int] | None:
    base_no = core.coerce_int(new.get("link"), 1)
    if base_no < 1 or base_no > core.MAX_LINK_NUMBER:
        _panel(
            "⛔ Link number invalid",
            [
                ("Reason", f"Link number {base_no} is outside 1..{core.MAX_LINK_NUMBER}."),
            ],
            kind="error",
        )
        _print_task(new)
        return None
    next_no = base_no + 1
    if next_no > core.MAX_LINK_NUMBER:
        _panel(
            "⛔ Link limit exceeded",
            [
                ("Reason", f"Link number {next_no} exceeds max_link_number={core.MAX_LINK_NUMBER}."),
            ],
            kind="error",
        )
        _print_task(new)
        return None
    return base_no, next_no


def _completion_kind_or_stop(new: dict, now_utc: datetime) -> str | None:
    raw_ch = (new.get("chain") or "").strip().lower()
    has_anchor = bool((new.get("anchor") or "").strip())
    has_cp = bool((new.get("cp") or "").strip())
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
            _print_task(new)
        return None

    kind = "anchor" if has_anchor else ("cp" if has_cp else None)
    if not kind:
        _print_task(new)
        return None
    return kind


def _completion_chain_id_or_fail(new: dict) -> str | None:
    chain_id = (new.get("chainID") or new.get("chainid") or "").strip()
    if chain_id:
        return chain_id
    _panel(
        "⛔ ChainID missing",
        [
            ("Reason", "ChainID is required in v3+ and legacy link-walk is removed."),
            ("Fix", "Run tools/nautical_backfill_chainid.py, then retry."),
        ],
        kind="error",
    )
    _print_task(new)
    return None


def _completion_existing_next_or_fail(new: dict, next_no: int) -> bool:
    existing_next = _existing_next_task(new, next_no)
    if not existing_next:
        return True
    ex_uuid = (existing_next.get("uuid") or "").strip()
    ex_short = _short(ex_uuid)
    ex_status = ((existing_next.get("status") or "").strip() or "unknown").lower()
    _panel(
        "ℹ Spawn skipped",
        [
            ("Reason", "Next link already exists for this completed task."),
            ("Existing", f"#{next_no} {ex_short} ({ex_status})"),
        ],
        kind="note",
    )
    _print_task(new)
    return False


def _completion_preflight_context(new: dict, now_utc: datetime) -> dict | None:
    parent_short = _short(new.get("uuid"))
    nums = _completion_link_numbers_or_fail(new)
    if nums is None:
        return None
    base_no, next_no = nums

    kind = _completion_kind_or_stop(new, now_utc)
    if not kind:
        return None

    chain_id = _completion_chain_id_or_fail(new)
    if not chain_id:
        return None

    if not _completion_existing_next_or_fail(new, next_no):
        return None

    return {
        "parent_short": parent_short,
        "base_no": base_no,
        "next_no": next_no,
        "kind": kind,
        "chain_id": chain_id,
    }


def _completion_compute_child_due(new: dict, kind: str):
    try:
        if kind == "anchor":
            child_due, meta, dnf = _compute_anchor_child_due(new)
        else:
            child_due, meta = _compute_cp_child_due(new)
            dnf = None
        return child_due, meta, dnf
    except ValueError as e:
        _panel(
            "⛔ Chain error",
            [("Reason", f"Invalid task field: {str(e)}")],
            kind="error",
        )
        _print_task(new)
        return None
    except Exception as e:
        _diag(f"compute next due failed: {e}")
        _panel(
            "⛔ Chain error",
            [("Reason", "Could not compute next due")],
            kind="error",
        )
        _print_task(new)
        return None


def _completion_until_or_fail(new: dict, now_utc: datetime) -> datetime | None | object:
    until_dt, err = _safe_parse_datetime(new.get("chainUntil"))
    if err:
        _panel(
            "⛔ Chain error", [("Reason", f"Invalid chainUntil: {err}")], kind="error"
        )
        _print_task(new)
        return False

    if until_dt:
        is_valid, err_msg = _validate_until_not_past(until_dt, now_utc)
        if not is_valid:
            _panel(
                "⛔ Chain error",
                [("Reason", f"Invalid chainUntil: {err_msg}")],
                kind="error",
            )
            _print_task(new)
            return False
    return until_dt


def _completion_until_guard_or_stop(new: dict, child_due, until_dt, now_utc: datetime) -> bool:
    if until_dt and child_due > until_dt:
        _end_chain_summary(new, "Reached 'until' limit", now_utc)
        new["chain"] = "off"
        _print_task(new)
        return False
    return True


def _completion_require_child_due_or_fail(new: dict, child_due) -> bool:
    if child_due:
        return True
    _panel(
        "⛔ Chain error",
        [("Reason", "Could not compute next due (no end date on parent)")],
        kind="error",
    )
    _print_task(new)
    return False


def _completion_warn_unreasonable_duration(new: dict, child_due, until_dt, now_utc: datetime) -> None:
    if not until_dt:
        return
    is_reasonable, warn_msg = _validate_chain_duration_reasonable(
        child_due, until_dt, now_utc
    )
    if warn_msg and not is_reasonable:
        _panel("⚠ Chain duration warning", [("Warning", warn_msg)], kind="warning")


def _completion_caps(kind: str, new: dict, child_due, dnf):
    cpmax = core.coerce_int(new.get("chainMax"), 0)
    until_dt = _dtparse(new.get("chainUntil"))
    cap_no = cpmax if cpmax else None
    finals = []

    if kind == "cp" and cpmax:
        try:
            fmax = _estimate_cp_final_by_max(new, child_due)
            if fmax:
                finals.append(("max", fmax))
        except Exception:
            pass
    if kind == "anchor" and cpmax:
        try:
            fmax = _estimate_anchor_final_by_max(new, child_due, dnf)
            if fmax:
                finals.append(("max", fmax))
        except Exception:
            pass

    until_cap_no = None
    if until_dt:
        if kind == "cp":
            u_no, u_dt = _cap_from_until_cp(new, child_due)
        else:
            u_no, u_dt = _cap_from_until_anchor(new, child_due, dnf)
        if u_no:
            until_cap_no = u_no
            cap_no = min(cap_no, u_no) if cap_no else u_no
        if u_dt:
            finals.append(("until", u_dt))
    return cpmax, until_dt, cap_no, finals, until_cap_no


def _completion_cap_guard_or_stop(new: dict, next_no: int, cap_no: int | None, now_utc: datetime) -> bool:
    if cap_no and next_no > cap_no:
        _end_chain_summary(new, f"Reached cap #{cap_no}", now_utc, current_task=new)
        new["chain"] = "off"
        _print_task(new)
        return False
    return True


def _completion_compute_next_and_limits(new: dict, kind: str, next_no: int, now_utc: datetime) -> dict | None:
    computed = _completion_compute_child_due(new, kind)
    if computed is None:
        return None
    child_due, meta, dnf = computed

    until_dt = _completion_until_or_fail(new, now_utc)
    if until_dt is False:
        return None

    if not _completion_until_guard_or_stop(new, child_due, until_dt, now_utc):
        return None

    if not _completion_require_child_due_or_fail(new, child_due):
        return None

    _completion_warn_unreasonable_duration(new, child_due, until_dt, now_utc)
    cpmax, until_dt, cap_no, finals, until_cap_no = _completion_caps(kind, new, child_due, dnf)

    if not _completion_cap_guard_or_stop(new, next_no, cap_no, now_utc):
        return None

    return {
        "child_due": child_due,
        "meta": meta,
        "dnf": dnf,
        "until_dt": until_dt,
        "cpmax": cpmax,
        "cap_no": cap_no,
        "finals": finals,
        "until_cap_no": until_cap_no,
    }


def _completion_build_and_spawn_child(
    new: dict,
    *,
    child_due,
    next_no: int,
    parent_short: str,
    kind: str,
    cpmax: int,
    until_dt,
) -> dict | None:
    # Build child payload & spawn
    try:
        child = _build_child_from_parent(
            new, child_due, next_no, parent_short, kind, cpmax, until_dt
        )
    except Exception as e:
        _diag(f"build child failed: {e}")
        _panel(
            "⛓ Chain error",
            [("Reason", "Failed to build next link")],
            kind="error",
        )
        _print_task(new)
        return None

    deferred_spawn = False
    spawn_intent_id = None
    try:
        (
            child_short,
            stripped_attrs,
            verified,
            deferred_spawn,
            defer_reason,
            spawn_intent_id,
        ) = _spawn_child_atomic(child, new)
        if not verified and not deferred_spawn:
            _panel(
                "⛓ Chain warning",
                [("Reason", defer_reason or "Child spawn could not be verified; parent not updated")],
                kind="warning",
            )
            _print_task(new)
            return None
    except Exception as e:
        _diag(f"spawn child failed: {e}")
        _panel(
            "⛓ Chain error",
            [("Reason", "Failed to spawn next link")],
            kind="error",
        )
        _print_task(new)
        return None

    # Reflect link on parent only when child is confirmed.
    if verified:
        new["nextLink"] = child_short

    return {
        "child": child,
        "child_short": child_short,
        "stripped_attrs": stripped_attrs,
        "verified": verified,
        "deferred_spawn": deferred_spawn,
        "spawn_intent_id": spawn_intent_id,
    }


def _handle_completion_modify(old: dict, new: dict) -> None:
    _completion_validate_cp_and_anchor(old, new)
    now_utc = core.now_utc()
    ctx = _completion_preflight_context(new, now_utc)
    if ctx is None:
        return
    parent_short = ctx["parent_short"]
    base_no = ctx["base_no"]
    next_no = ctx["next_no"]
    kind = ctx["kind"]
    chain_id = ctx["chain_id"]

    computed = _completion_compute_next_and_limits(new, kind, next_no, now_utc)
    if computed is None:
        return
    child_due = computed["child_due"]
    meta = computed["meta"]
    dnf = computed["dnf"]
    until_dt = computed["until_dt"]
    cpmax = computed["cpmax"]
    cap_no = computed["cap_no"]
    finals = computed["finals"]
    until_cap_no = computed["until_cap_no"]

    spawned = _completion_build_and_spawn_child(
        new,
        child_due=child_due,
        next_no=next_no,
        parent_short=parent_short,
        kind=kind,
        cpmax=cpmax,
        until_dt=until_dt,
    )
    if spawned is None:
        return
    child = spawned["child"]
    child_short = spawned["child_short"]
    stripped_attrs = spawned["stripped_attrs"]
    deferred_spawn = spawned["deferred_spawn"]
    spawn_intent_id = spawned["spawn_intent_id"]

    # Build an in-memory chain index once for panel/timeline lookups.
    chain = []
    chain_by_link = None
    chain_by_short = None
    chain_id = (new.get("chainID") or new.get("chainid") or "").strip()
    need_chain = _SHOW_ANALYTICS or _SHOW_TIMELINE_GAPS or _CHECK_CHAIN_INTEGRITY
    if chain_id and need_chain:
        try:
            chain = _get_chain_export(chain_id)
            if chain:
                chain_by_link, chain_by_short = _build_chain_indexes(chain)
                _set_chain_cache(chain_id, chain)
                _export_uuid_short_cached.cache_clear()
        except Exception:
            pass
    global _PANEL_CHAIN_BY_LINK
    _PANEL_CHAIN_BY_LINK = chain_by_link
    global _PANEL_CHAIN_BY_SHORT
    _PANEL_CHAIN_BY_SHORT = chain_by_short
    analytics_advice = None
    integrity_warnings = None
    if _SHOW_ANALYTICS and chain:
        try:
            analytics_advice = _chain_health_advice(chain, kind, new, style=_ANALYTICS_STYLE)
        except Exception:
            analytics_advice = None
    if _CHECK_CHAIN_INTEGRITY and chain:
        try:
            integrity_warnings = _chain_integrity_warnings(chain, expected_chain_id=chain_id)
        except Exception:
            integrity_warnings = None

    if kind == "anchor":
        _render_anchor_completion_feedback(
            new=new,
            child=child,
            child_due=child_due,
            child_short=child_short,
            next_no=next_no,
            parent_short=parent_short,
            cap_no=cap_no,
            finals=finals,
            now_utc=now_utc,
            until_dt=until_dt,
            until_cap_no=until_cap_no,
            dnf=dnf,
            meta=meta,
            stripped_attrs=stripped_attrs,
            deferred_spawn=deferred_spawn,
            spawn_intent_id=spawn_intent_id,
            chain_by_short=chain_by_short,
            analytics_advice=analytics_advice,
            integrity_warnings=integrity_warnings,
            base_no=base_no,
        )
    else:
        _render_cp_completion_feedback(
            new=new,
            child=child,
            child_due=child_due,
            child_short=child_short,
            next_no=next_no,
            parent_short=parent_short,
            cap_no=cap_no,
            finals=finals,
            now_utc=now_utc,
            until_dt=until_dt,
            until_cap_no=until_cap_no,
            meta=meta,
            deferred_spawn=deferred_spawn,
            spawn_intent_id=spawn_intent_id,
            chain_by_short=chain_by_short,
            analytics_advice=analytics_advice,
            integrity_warnings=integrity_warnings,
            base_no=base_no,
        )




    _print_task(new)
    _diag_summary()


def main():
    old, new = _read_two()

    # Skip all Nautical logic when task is being deleted
    if (new.get("status") or "").lower() == "deleted":
        print(json.dumps(new, ensure_ascii=False))
        return

    if not _task_has_nautical_fields(old, new):
        print(json.dumps(new, ensure_ascii=False))
        return

    try:
        _load_core()
    except Exception as e:
        _diag(f"core load failed: {e}")
        _fail_and_exit("Hook misconfigured", "Failed to initialize nautical core")

    if _is_non_completion_modify(old, new):
        _handle_non_completion_modify(old, new)
        return

    _handle_completion_modify(old, new)


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        if os.environ.get("NAUTICAL_DIAG") == "1":
            try:
                sys.stderr.write(f"[nautical] on-modify unexpected error: {e}\n")
            except Exception:
                pass
        _panic_passthrough()
        raise SystemExit(1)

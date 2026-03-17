#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared core for Taskwarrior Nautical hooks.

"""
from __future__ import annotations
import os, re, sys
import copy
import math
import stat
from collections import OrderedDict
from types import MappingProxyType
from typing import Any, TypeAlias, TypedDict, cast
from datetime import datetime, timedelta, timezone, date
from functools import lru_cache, wraps
from calendar import month_name, monthrange
from datetime import date as _date
import json, zlib, base64, hashlib, tempfile, time, random, subprocess
import difflib
from contextlib import contextmanager
try:
    import fcntl  # POSIX advisory lock
except Exception:
    fcntl = None


class AnchorMods(TypedDict, total=False):
    t: str
    bd: bool
    wd: bool
    pbd: int
    nbd: int
    nw: bool


class AnchorAtom(TypedDict, total=False):
    typ: str
    type: str
    spec: str
    value: str
    interval: int
    mods: AnchorMods


AnchorTerm: TypeAlias = list[AnchorAtom]
AnchorDNF: TypeAlias = list[AnchorTerm]
TaskDict: TypeAlias = dict[str, Any]
AnchorValidationResult: TypeAlias = tuple[AnchorDNF | None, str | None]


class HintMetaCfg(TypedDict, total=False):
    fmt: str
    salt: str
    tz: str
    hol: str


class HintMeta(TypedDict, total=False):
    created: int
    cfg: HintMetaCfg


class HintPerYear(TypedDict, total=False):
    est: int
    first: str
    last: str


class HintLimits(TypedDict, total=False):
    stop: str
    max_left: int
    until: str


class AnchorHintsPayload(TypedDict, total=False):
    meta: HintMeta
    dnf: AnchorDNF
    natural: str
    next_dates: list[str]
    per_year: HintPerYear
    limits: HintLimits
    rand_preview: list[str]


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
    tomllib = None
if tomllib is None:
    try:
        import tomli

        tomllib = tomli  # Python 3.10 and earlier (pip install tomli)
    except Exception:
        tomllib = None


# --- Defaults ---
_DEFAULTS = {
    "wrand_salt": "nautical|wrand|v3",  # change to reshuffle weekly-rand streams
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
_CACHE_LOAD_MEM_MAX = 128
_CACHE_LOAD_MEM_TTL = 300
_CACHE_LOAD_MEM: OrderedDict[str, tuple[int, int, dict, float]] = OrderedDict()

def _env_flag_true(name: str, env_map: dict | None = None) -> bool:
    src = env_map if env_map is not None else os.environ
    try:
        raw = src.get(name, "") if hasattr(src, "get") else ""
    except Exception:
        raw = ""
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _path_input_error(path_value: str) -> str | None:
    raw = str(path_value or "").strip()
    if not raw:
        return "empty path"
    if "\x00" in raw:
        return "NUL byte in path"
    # Block traversal segments before normalization.
    parts = raw.replace("\\", "/").split("/")
    if any(p == ".." for p in parts):
        return "parent traversal ('..') is not allowed"
    return None


def _normalized_abspath(path_value: str) -> str:
    return os.path.abspath(os.path.expanduser(str(path_value or "").strip()))


def _nearest_existing_dir(path_value: str) -> str | None:
    cur = _normalized_abspath(path_value)
    while cur:
        if os.path.isdir(cur):
            return cur
        parent = os.path.dirname(cur)
        if not parent or parent == cur:
            return None
        cur = parent
    return None


def _world_writable_without_sticky(mode: int) -> bool:
    return bool(mode & 0o002) and not bool(mode & 0o1000)


def _path_safety_error(path_value: str, *, expect_dir: bool = True) -> str | None:
    """
    Return a short error string when path is unsafe for filesystem operations.

    Rules:
      - Existing target dirs must be owned by current user (when getuid is available).
      - Existing target/ancestor must not be world-writable unless sticky bit is set.
      - Path must resolve to / under a writable+searchable existing ancestor.
    """
    ap = _normalized_abspath(path_value)
    if not ap:
        return "empty path"
    try:
        target_exists = os.path.exists(ap)
        if target_exists:
            if expect_dir and not os.path.isdir(ap):
                return "not a directory"
            probe = ap
        else:
            probe = _nearest_existing_dir(ap)
            if not probe:
                return "no existing parent directory"

        st = os.stat(probe)
        if target_exists:
            uid_fn = getattr(os, "getuid", None)
            if callable(uid_fn):
                uid = uid_fn()
                if st.st_uid != uid:
                    return "owner mismatch"
        if _world_writable_without_sticky(st.st_mode):
            return "world-writable path without sticky bit"
        if expect_dir:
            if not os.access(probe, os.W_OK | os.X_OK):
                return "path is not writable/searchable"
        else:
            if target_exists and not os.path.isdir(ap):
                if not os.access(ap, os.R_OK):
                    return "path is not readable"
            elif not os.access(probe, os.W_OK | os.X_OK):
                return "parent path is not writable/searchable"
    except Exception as e:
        return str(e)
    return None


def _validated_user_dir(
    path_value: str,
    *,
    label: str,
    trust_env: str = "",
    env_map: dict | None = None,
    warn_on_error: bool = True,
) -> str:
    raw = str(path_value or "").strip()
    in_err = _path_input_error(raw)
    if in_err:
        if warn_on_error and _env_flag_true("NAUTICAL_DIAG", env_map):
            try:
                sys.stderr.write(f"[nautical] Ignoring unsafe {label} '{raw}': {in_err}\n")
            except Exception:
                pass
        return ""
    ap = _normalized_abspath(raw)
    if trust_env and _env_flag_true(trust_env, env_map):
        return ap
    err = _path_safety_error(ap, expect_dir=True)
    if err:
        if warn_on_error and _env_flag_true("NAUTICAL_DIAG", env_map):
            try:
                sys.stderr.write(f"[nautical] Ignoring unsafe {label} '{path_value}': {err}\n")
            except Exception:
                pass
        return ""
    return ap


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
    trust_config_path = _env_flag_true("NAUTICAL_TRUST_CONFIG_PATH")

    if not trust_config_path:
        in_err = _path_input_error(path)
        if in_err:
            if os.environ.get("NAUTICAL_DIAG") == "1":
                try:
                    sys.stderr.write(f"[nautical] Ignoring unsafe config path '{path}': {in_err}\n")
                except Exception:
                    pass
            return {}
        safety_err = _path_safety_error(path, expect_dir=False)
        if safety_err:
            if os.environ.get("NAUTICAL_DIAG") == "1":
                try:
                    sys.stderr.write(f"[nautical] Ignoring unsafe config path '{path}': {safety_err}\n")
                except Exception:
                    pass
            return {}

    # File exists, but cannot parse TOML (Python < 3.11 and no tomli)
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
        raw_env = str(env_path).strip()
        in_err = _path_input_error(raw_env)
        if in_err and not _env_flag_true("NAUTICAL_TRUST_CONFIG_PATH"):
            if os.environ.get("NAUTICAL_DIAG") == "1":
                try:
                    sys.stderr.write(f"[nautical] Ignoring unsafe NAUTICAL_CONFIG '{raw_env}': {in_err}\n")
                except Exception:
                    pass
            return []
        ap = os.path.abspath(os.path.expanduser(raw_env))
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
    # Alias support:
    #   recurrence_update_udas = [...]
    #   [recurrence] update_udas = [...]
    #   recurrence.update_udas = "..."
    if cfg.get("recurrence_update_udas") is None:
        rec = cfg.get("recurrence")
        if isinstance(rec, dict):
            rec_norm = _normalize_keys(rec)
            if rec_norm.get("update_udas") is not None:
                cfg["recurrence_update_udas"] = rec_norm.get("update_udas")
    if cfg.get("recurrence_update_udas") is None and cfg.get("recurrence.update_udas") is not None:
        cfg["recurrence_update_udas"] = cfg.get("recurrence.update_udas")
    return cfg



def _nautical_cache_dir() -> str:
    from . import cache_support as _cache_support

    return _cache_support.nautical_cache_dir(validated_user_dir=_validated_user_dir)


def _warn_once_per_day(key: str, message: str) -> None:
    from . import warnings as _warnings

    _warnings.warn_once_per_day(
        key,
        message,
        cache_dir=_nautical_cache_dir(),
        require_diag=True,
    )


def _warn_once_per_day_any(key: str, message: str) -> None:
    from . import warnings as _warnings

    _warnings.warn_once_per_day(
        key,
        message,
        cache_dir=_nautical_cache_dir(),
        require_diag=False,
    )


def _warn_rate_limited_any(key: str, message: str, min_interval_s: float = 3600.0) -> None:
    from . import warnings as _warnings

    _warnings.warn_rate_limited_any(
        key,
        message,
        cache_dir=_nautical_cache_dir(),
        min_interval_s=min_interval_s,
    )


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
        _CACHE_LOAD_MEM.clear()
    except Exception:
        pass
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
    try:
        _cache_key_for_task_cached.cache_clear()
    except Exception:
        pass


# -------- UI helpers ----------------------------------------------------------
from . import ui as _ui

strip_rich_markup = _ui.strip_rich_markup
term_width_stderr = _ui.term_width_stderr
fast_color_enabled = _ui.fast_color_enabled
ansi = _ui.ansi
emit_wrapped = _ui.emit_wrapped
emit_line = _ui.emit_line
panel_line_from_rows = _ui.panel_line_from_rows
panel_line = _ui.panel_line
render_panel = _ui.render_panel



def _warn_missing_toml_parser(config_path: str) -> None:
    from . import warnings as _warnings

    _warnings.warn_missing_toml_parser(
        config_path,
        warn_once_per_day=_warn_once_per_day,
        warn_once_per_day_any=_warn_once_per_day_any,
    )


def _warn_toml_parse_error(config_path: str, err: Exception) -> None:
    from . import warnings as _warnings

    _warnings.warn_toml_parse_error(
        config_path,
        err,
        warn_once_per_day=_warn_once_per_day,
        warn_once_per_day_any=_warn_once_per_day_any,
    )


def _get_config() -> dict:
    global _CONF_CACHE
    if _CONF_CACHE is None:
        # Cache an internal mutable copy, but never expose it directly.
        _CONF_CACHE = copy.deepcopy(_load_config())
    return copy.deepcopy(_CONF_CACHE)

_CONF = MappingProxyType(_get_config())

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


def _conf_csv_or_list(key: str, default: list[str] | None = None, lower: bool = False) -> list[str]:
    v = _conf_raw(key)
    if v is None:
        return list(default or [])
    if isinstance(v, str):
        raw_items = v.split(",")
    elif isinstance(v, (list, tuple, set)):
        raw_items = list(v)
    else:
        raw_items = [v]

    out: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        s = str(item).strip()
        if not s:
            continue
        if lower:
            s = s.lower()
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out if out else list(default or [])


_UDA_ATTR_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def _conf_uda_field_list(key: str) -> list[str]:
    fields = _conf_csv_or_list(key, default=[], lower=True)
    out: list[str] = []
    for f in fields:
        if _UDA_ATTR_NAME_RE.fullmatch(f):
            out.append(f)
            continue
        if os.environ.get("NAUTICAL_DIAG") == "1":
            try:
                print(f"[nautical] Ignoring invalid UDA field in {key}: {f!r}", file=sys.stderr)
            except Exception:
                pass
    return out

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
RECURRENCE_UPDATE_UDAS = tuple(_conf_uda_field_list("recurrence_update_udas"))
_CACHE_TTL_SECS = _conf_int("cache_ttl_secs", 3600, min_value=0)
_CACHE_LOAD_MEM_MAX = _conf_int("cache_load_mem_max", _CACHE_LOAD_MEM_MAX, min_value=16, max_value=4096)
_CACHE_LOAD_MEM_TTL = _conf_int("cache_load_mem_ttl", _CACHE_LOAD_MEM_TTL, min_value=0, max_value=86400)

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
from . import common as _common
from . import nth_monthly as _nth_monthly
from . import quarter_helpers as _quarter_helpers
from . import quarter_rewrite as _quarter_rewrite
from . import quarter_selector as _quarter_selector
from . import tokenutil as _tokenutil
from . import year_tokens as _year_tokens

short_uuid = _common.short_uuid
should_stamp_chain_id = _common.should_stamp_chain_id

# ==============================================================================
# SECTION: Time & timezone helpers
# ==============================================================================
try:
    import zoneinfo as _zoneinfo
except Exception:
    _zoneinfo = None

if _zoneinfo is None:
    _LOCAL_TZ = None
    _warn_once_per_day(
        "timezone_zoneinfo_unavailable",
        "[nautical] timezone support unavailable (zoneinfo import failed); using UTC fallback.",
    )
else:
    try:
        _LOCAL_TZ = _zoneinfo.ZoneInfo(LOCAL_TZ_NAME)
    except Exception:
        _LOCAL_TZ = None
        _warn_once_per_day(
            "timezone_local_invalid",
            f"[nautical] timezone '{LOCAL_TZ_NAME}' is invalid/unavailable; using UTC fallback.",
        )

from . import timeutil as _timeutil

def now_utc():
    return _timeutil.now_utc()


def to_local(dt_utc: datetime) -> datetime:
    return _timeutil.to_local(dt_utc, _LOCAL_TZ)


def fmt_dt_local(dt_utc: datetime) -> str:
    return _timeutil.fmt_dt_local(dt_utc, _LOCAL_TZ)


def fmt_isoz(dt_utc: datetime) -> str:
    return _timeutil.fmt_isoz(dt_utc)


def _ensure_utc(dt_utc: datetime) -> datetime:
    return _timeutil.ensure_utc(dt_utc)


# --- Date/time config ---
DATE_FORMATS = ("%Y%m%dT%H%M%SZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d")
UNTIL_COUNT_CAP = 1000
INTERSECTION_GUARD_STEPS = 256
DEFAULT_DUE_HOUR = 11
MAX_ANCHOR_DNF_TERMS = _conf_int("max_anchor_dnf_terms", 10_000, min_value=64, max_value=200_000)

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

_MONTH_ALIAS = _tokenutil.MONTH_ALIAS



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
_int_floatish_re = _common._INT_FLOATISH_RE
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
_CONTROL_CHARS_RE = _common._CONTROL_CHARS_RE

def _safe_match(pattern: re.Pattern, text: str, max_len: int = 256):
    """Defensive regex match to avoid pathological backtracking."""
    if text is None:
        return None
    if len(text) > max_len:
        raise ParseError("Expression too complex")
    return pattern.match(text)


def sanitize_text(v: str, max_len: int = 1024) -> str:
    return _common.sanitize_text(v, max_len=max_len)


def sanitize_task_strings(task: dict, max_len: int = 1024) -> None:
    _common.sanitize_task_strings(task, max_len=max_len)


def _split_csv_tokens(spec: str) -> list[str]:
    return _common.split_csv_tokens(spec)


def _split_csv_lower(spec: str) -> list[str]:
    return _common.split_csv_lower(spec)


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
    return _tokenutil.static_month_last_day(mm)


def _month_from_alias(tok: str) -> int | None:
    return _tokenutil.month_from_alias(tok)


def _year_full_months_span_token(m1: int, m2: int) -> str:
    return _year_tokens.year_full_months_span_token(m1, m2, tok_range=_tok_range)


def _rewrite_month_names_to_ranges(spec: str) -> str:
    return _year_tokens.rewrite_month_names_to_ranges(spec, tok_range=_tok_range)


# --- helpers for nth-weekday monthly /N gating ---
def _parse_nth_wd_tokens(spec: str):
    return _nth_monthly.parse_nth_wd_tokens(
        spec,
        split_csv_lower=_split_csv_lower,
        nth_weekday_re=_nth_weekday_re,
        weekdays=_WEEKDAYS,
    )


def _month_has_any_nth(y: int, m: int, pairs: list[tuple[int, int]]) -> bool:
    return _nth_monthly.month_has_any_nth(y, m, pairs, month_len=month_len)


def _advance_to_next_allowed_month(y: int, m: int, pairs) -> tuple[int, int]:
    return _nth_monthly.advance_to_next_allowed_month(
        y,
        m,
        pairs,
        month_has_any_nth=_month_has_any_nth,
    )

def _unwrap_quotes(s: str) -> str:
    return _tokenutil.unwrap_quotes(s)

def _year_full_month_range_token(mm: int) -> str:
    return _year_tokens.year_full_month_range_token(mm, tok_range=_tok_range)

def _mon_to_int(tok: str) -> int | None:
    return _tokenutil.mon_to_int(tok)


def _rewrite_year_month_aliases_in_context(dnf: list[list[dict]]) -> list[list[dict]]:
    return _year_tokens.rewrite_year_month_aliases_in_context(dnf, tok_range=_tok_range)

# --- Anchor Canonical Form (ACF) ----------------------------------------------


# Constants - no runtime config dependency
ACF_COMPRESSED = True  # Always compress for best storage
ACF_CHECKSUM_LEN = 8   # 8 chars = 32 bits of entropy


# Weekday normalize, with rand / rand*
_WD_ABBR = _tokenutil.WD_ABBR
_WEEKLY_ALIAS = _tokenutil.WEEKLY_ALIAS
_MONTHLY_ALIAS = _tokenutil.MONTHLY_ALIAS

def _expand_weekly_aliases(spec: str) -> str:
    return _tokenutil.expand_weekly_aliases(spec)


# ==============================================================================
# SECTION: Anchor parsing (DNF/ACF helpers)
# ==============================================================================
# --- Token normalization ---
def _expand_monthly_aliases(spec: str) -> str:
    return _tokenutil.expand_monthly_aliases(spec)
def _normalize_weekday(s: str) -> str | None:
    return _tokenutil.normalize_weekday(s)

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
            ival = int(coerce_int(a.get("ival"), 1) or 1)
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
        return _clone_mod_value(res)
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
    out: dict[str, object] = {}
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
    from . import cache_support as _cache_support

    chosen = _cache_support.select_cache_dir(
        anchor_cache_dir_override=ANCHOR_CACHE_DIR_OVERRIDE,
        nautical_cache_dir_path=_nautical_cache_dir(),
        validated_user_dir=_validated_user_dir,
    )
    _CACHE_DIR = chosen
    return chosen

def _cache_key(acf: str, anchor_mode: str) -> str:
    from . import cache_support as _cache_support

    return _cache_support.cache_key(
        acf,
        anchor_mode,
        anchor_year_fmt=ANCHOR_YEAR_FMT,
        wrand_salt=WRAND_SALT,
        local_tz_name=LOCAL_TZ_NAME,
        holiday_region=HOLIDAY_REGION,
    )

def _cache_path(key: str) -> str:
    from . import cache_support as _cache_support

    return _cache_support.cache_path(_cache_dir(), key)

def _cache_lock_path(key: str) -> str:
    from . import cache_support as _cache_support

    return _cache_support.cache_lock_path(_cache_dir(), key)

@contextmanager
def _safe_lock_sleep_once(sleep_base: float, jitter: float) -> None:
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


def _safe_lock_ensure_parent(path_str: str, mkdir: bool) -> None:
    if not mkdir:
        return
    try:
        parent = os.path.dirname(path_str)
        if parent:
            os.makedirs(parent, exist_ok=True)
    except Exception:
        pass


def _safe_lock_age(path_str: str) -> float | None:
    try:
        with open(path_str, "r", encoding="utf-8") as f:
            head = f.read(64)
        parts = head.strip().split()
        if len(parts) >= 2:
            return time.time() - float(parts[1])
    except Exception:
        pass
    try:
        st = os.stat(path_str)
        return time.time() - float(st.st_mtime)
    except Exception:
        return None


def _safe_lock_stale_pid(path_str: str, stale_after: float | None) -> bool:
    try:
        with open(path_str, "r", encoding="utf-8") as f:
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


@contextmanager
def _safe_lock_fcntl_context(
    path_str: str,
    *,
    tries: int,
    sleep_base: float,
    jitter: float,
    mode: int,
    mkdir: bool,
):
    lf = None
    acquired = False
    _safe_lock_ensure_parent(path_str, mkdir)
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
                _safe_lock_sleep_once(sleep_base, jitter)
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


@contextmanager
def _safe_lock_excl_context(
    path_str: str,
    *,
    tries: int,
    sleep_base: float,
    jitter: float,
    mode: int,
    mkdir: bool,
    stale_after: float | None,
):
    fd = None
    acquired = False
    for _ in range(tries):
        _safe_lock_ensure_parent(path_str, mkdir)
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
            pid_stale = _safe_lock_stale_pid(path_str, stale_after)
            age_stale = False
            if stale_after is not None:
                age = _safe_lock_age(path_str)
                if age is not None and age >= float(stale_after):
                    age_stale = True
            if pid_stale and age_stale:
                try:
                    os.unlink(path_str)
                except Exception:
                    pass
            else:
                _safe_lock_sleep_once(sleep_base, jitter)
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

    tries = max(1, int(retries or 0))

    if fcntl is not None:
        with _safe_lock_fcntl_context(
            path_str,
            tries=tries,
            sleep_base=sleep_base,
            jitter=jitter,
            mode=mode,
            mkdir=mkdir,
        ) as acquired:
            yield acquired
        return

    with _safe_lock_excl_context(
        path_str,
        tries=tries,
        sleep_base=sleep_base,
        jitter=jitter,
        mode=mode,
        mkdir=mkdir,
        stale_after=stale_after,
    ) as acquired:
        yield acquired

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
from . import runtime as _runtime

_DIAG_LOG_REDACT_KEYS = _runtime.DIAG_LOG_REDACT_KEYS
_hook_arg_value = _runtime.hook_arg_value
resolve_task_data_context = _runtime.resolve_task_data_context
diag_log_redact = _runtime.diag_log_redact
diag_log = _runtime.diag_log
diag = _runtime.diag
_run_task_should_retry = _runtime._run_task_should_retry
_run_task_retry_sleep = _runtime._run_task_retry_sleep
_run_task_prepare_tempfiles = _runtime._run_task_prepare_tempfiles
_run_task_normalize_input = _runtime._run_task_normalize_input
_run_task_collect_outputs = _runtime._run_task_collect_outputs
_run_task_cleanup_paths = _runtime._run_task_cleanup_paths
run_task = _runtime.run_task
is_lock_error = _runtime.is_lock_error


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


def _clone_mod_value(v):
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, list):
        out = []
        for item in v:
            if isinstance(item, tuple):
                out.append((item[0], item[1]) if len(item) == 2 else tuple(item))
            elif isinstance(item, list):
                out.append((item[0], item[1]) if len(item) == 2 else list(item))
            else:
                out.append(item)
        return out
    if isinstance(v, tuple):
        return (v[0], v[1]) if len(v) == 2 else tuple(v)
    if isinstance(v, dict):
        return {k: _clone_mod_value(val) for k, val in v.items()}
    return v


def _clone_mods(mods):
    if not isinstance(mods, dict):
        return {}
    out = {}
    for k, v in mods.items():
        if v is None or isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif k == "t" and isinstance(v, tuple):
            out[k] = (v[0], v[1]) if len(v) == 2 else tuple(v)
        elif k == "t" and isinstance(v, list):
            tv = []
            for item in v:
                if isinstance(item, tuple):
                    tv.append((item[0], item[1]) if len(item) == 2 else tuple(item))
                elif isinstance(item, list):
                    tv.append((item[0], item[1]) if len(item) == 2 else list(item))
                else:
                    tv.append(item)
            out[k] = tv
        else:
            out[k] = _clone_mod_value(v)
    return out


def _clone_atom(atom):
    if not isinstance(atom, dict):
        return atom
    out = dict(atom)
    mods = atom.get("mods")
    if isinstance(mods, dict):
        out["mods"] = _clone_mods(mods)
    qmap = atom.get("_qmap")
    if isinstance(qmap, dict):
        out["_qmap"] = dict(qmap)
    for k, v in atom.items():
        if k in ("mods", "_qmap"):
            continue
        if isinstance(v, (dict, list, tuple)):
            out[k] = _clone_mod_value(v)
    return out


def _clone_dnf(dnf):
    if not isinstance(dnf, (list, tuple)):
        return dnf
    out = []
    for term in dnf:
        if isinstance(term, (list, tuple)):
            out.append([_clone_atom(atom) for atom in term])
        else:
            out.append(term)
    return out


def _clone_cache_payload(obj: dict) -> dict:
    if not isinstance(obj, dict):
        return obj
    out = {}
    for k, v in obj.items():
        if k == "dnf":
            out[k] = _clone_dnf(v)
        elif isinstance(v, list):
            out[k] = list(v)
        elif isinstance(v, dict):
            inner = {}
            for ik, iv in v.items():
                if isinstance(iv, (dict, list, tuple)):
                    inner[ik] = _clone_mod_value(iv)
                else:
                    inner[ik] = iv
            out[k] = inner
        elif isinstance(v, (dict, list, tuple)):
            out[k] = _clone_mod_value(v)
        else:
            out[k] = v
    return out


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


def _cache_payload_shape_ok(obj: dict) -> bool:
    """Validate cached payload shape defensively to avoid downstream type errors."""
    try:
        if "dnf" in obj and not _is_dnf_like(obj.get("dnf")):
            return False
        natural = obj.get("natural")
        if natural is not None and not isinstance(natural, str):
            return False
        next_dates = obj.get("next_dates")
        if next_dates is not None:
            if not isinstance(next_dates, list):
                return False
            for item in next_dates:
                if not isinstance(item, str):
                    return False
        meta = obj.get("meta")
        if meta is not None and not isinstance(meta, dict):
            return False
        per_year = obj.get("per_year")
        if per_year is not None and not isinstance(per_year, dict):
            return False
        limits = obj.get("limits")
        if limits is not None and not isinstance(limits, dict):
            return False
    except Exception:
        return False
    return True


def _cache_atomic_replace(src: str, dst: str) -> None:
    """Best-effort atomic replace across platforms."""
    try:
        os.replace(src, dst)
        return
    except OSError:
        if os.name != "nt":
            raise
    try:
        import ctypes

        flags = 0x1 | 0x8  # MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH
        ok = ctypes.windll.kernel32.MoveFileExW(str(src), str(dst), flags)
        if ok:
            return
        err = ctypes.GetLastError()
        raise OSError(err, "MoveFileExW failed")
    except Exception:
        raise

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
        stamp = (int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000))), int(st.st_size))
        now = time.time()
        if _CACHE_LOAD_MEM_TTL > 0 and _CACHE_LOAD_MEM:
            expired = [
                k for k, (_mt, _sz, _obj, loaded_at) in _CACHE_LOAD_MEM.items()
                if (now - loaded_at) > _CACHE_LOAD_MEM_TTL
            ]
            for k in expired:
                _CACHE_LOAD_MEM.pop(k, None)
        memo = _CACHE_LOAD_MEM.get(key)
        if memo and memo[0] == stamp[0] and memo[1] == stamp[1]:
            if _CACHE_LOAD_MEM_TTL <= 0 or (now - memo[3]) <= _CACHE_LOAD_MEM_TTL:
                _CACHE_LOAD_MEM.move_to_end(key)
                return _clone_cache_payload(memo[2])
            _CACHE_LOAD_MEM.pop(key, None)
        with open(path, "rb") as f:
            blob = f.read()
        data = zlib.decompress(base64.b85decode(blob))
        obj = json.loads(data.decode("utf-8"))
        if isinstance(obj, dict) and "dnf" in obj:
            obj["dnf"] = _normalize_dnf_cached(obj.get("dnf"))
        if isinstance(obj, dict):
            if not _cache_payload_shape_ok(obj):
                if os.environ.get("NAUTICAL_DIAG") == "1":
                    diag(f"cache_load rejected invalid payload shape for key={key}")
                return None
            _CACHE_LOAD_MEM[key] = (stamp[0], stamp[1], obj, now)
            _CACHE_LOAD_MEM.move_to_end(key)
            if len(_CACHE_LOAD_MEM) > _CACHE_LOAD_MEM_MAX:
                _CACHE_LOAD_MEM.popitem(last=False)
            return _clone_cache_payload(obj)
        return None
    except (OSError, ValueError, json.JSONDecodeError, zlib.error) as e:
        if os.environ.get("NAUTICAL_DIAG") == "1":
            diag(f"cache_load failed: {e}")
        return None

def cache_save(key: str, obj: dict) -> bool:
    if not ENABLE_ANCHOR_CACHE:
        return False
    data = json.dumps(obj, separators=(",", ":"), sort_keys=True).encode("utf-8")
    blob = base64.b85encode(zlib.compress(data, 9))
    path = _cache_path(key)
    if not path:
        return False
    tmpf = None
    ok_saved = False
    try:
        base = _cache_dir()
        if not base:
            return False
        with _cache_lock(key) as locked:
            if not locked:
                if os.environ.get("NAUTICAL_DIAG") == "1":
                    diag(f"cache_save lock busy for key={key}")
                return False
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
            _cache_atomic_replace(tmpf, path)
            ok_saved = True
    except (OSError, ValueError, json.JSONDecodeError, zlib.error) as e:
        if os.environ.get("NAUTICAL_DIAG") == "1":
            diag(f"cache_save failed: {e}")
    finally:
        _CACHE_LOAD_MEM.pop(key, None)
        if tmpf and os.path.exists(tmpf):
            try:
                os.unlink(tmpf)
            except Exception:
                pass
    return ok_saved

@_ttl_lru_cache(maxsize=1024)
def _cache_key_for_task_cached(anchor_expr: str, anchor_mode: str, fmt: str) -> str:
    _ = fmt  # cache key dimension; keeps keys correct across DM/MD flips
    try:
        acf = build_acf(anchor_expr)
    except Exception:
        acf = (anchor_expr or "").strip()
    return _cache_key(acf, anchor_mode or "")


def cache_key_for_task(anchor_expr: str, anchor_mode: str) -> str:
    return _cache_key_for_task_cached(anchor_expr or "", anchor_mode or "", _yearfmt())


# ---- Core iterator over DNF ---------------------------------------------------
_NTH_RE  = re.compile(r"^(?:(\d)(?:st|nd|rd|th)|last)-(" + "|".join(_WD_ABBR) + r")$")

def _days_in_month(y:int, m:int) -> int:
    return monthrange(y, m)[1]

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
    allowed: set[int] = set()
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


def _term_has_monthly_rand(term: list[dict]) -> bool:
    return any(
        (a.get("typ") or a.get("type")) == "m"
        and "rand" in str(a.get("spec") or "").lower()
        for a in term
    )


def _term_year_specs(term: list[dict]) -> list[str]:
    return [str(a.get("spec") or "") for a in term if (a.get("typ") or a.get("type")) == "y"]


def _first_day_next_month(y: int, m: int) -> date:
    return date(y, m, 1) + timedelta(days=_days_in_month(y, m))


def _month_allowed_doms_for_monthly_atom(atom: dict, y: int, m: int, dim: int) -> set[int]:
    spec = str(atom.get("spec") or "")
    toks = _split_csv_lower(spec)
    if not toks:
        return set(range(1, dim + 1))
    doms: set[int] = set()
    for tok in toks:
        if tok == "rand":
            doms.update(range(1, dim + 1))
        else:
            doms.update(_doms_for_monthly_token(tok, y, m))
    return doms


def _intersect_monthly_atoms_allowed(
    term: list[dict],
    *,
    y: int,
    m: int,
    dim: int,
    allowed: set[int],
) -> set[int]:
    out = set(allowed)
    for atom in term:
        typ = (atom.get("typ") or atom.get("type") or "").lower()
        if typ != "m":
            continue
        out &= _month_allowed_doms_for_monthly_atom(atom, y, m, dim)
        if not out:
            return set()
    return out


def _intersect_weekly_atoms_allowed(
    term: list[dict],
    *,
    y: int,
    m: int,
    allowed: set[int],
) -> set[int]:
    out = set(allowed)
    for atom in term:
        typ = (atom.get("typ") or atom.get("type") or "").lower()
        if typ != "w":
            continue
        spec = str(atom.get("spec") or "")
        wdom = _doms_for_weekly_spec(spec, y, m)
        out = out & wdom if wdom else set()
        if not out:
            return set()
    return out


def _next_for_and_rand_yearly(term: list[dict], ref_d: date, y_specs: list[str]) -> date | None:
    probe = ref_d + timedelta(days=1)
    for _ in range(60):  # scan up to 5 years (60 months)
        y, m = probe.year, probe.month
        dim = _days_in_month(y, m)
        allowed = set(range(1, dim + 1))
        allowed &= _doms_allowed_by_year(y, m, y_specs)
        if not allowed:
            probe = _first_day_next_month(y, m)
            continue

        allowed = _intersect_monthly_atoms_allowed(term, y=y, m=m, dim=dim, allowed=allowed)
        if not allowed:
            probe = _first_day_next_month(y, m)
            continue

        allowed = _intersect_weekly_atoms_allowed(term, y=y, m=m, allowed=allowed)
        if not allowed:
            probe = _first_day_next_month(y, m)
            continue

        pick = _choose_rand_dom(y, m, allowed)
        if pick is None:
            probe = _first_day_next_month(y, m)
            continue
        cand = date(y, m, pick)
        if cand > ref_d:
            return cand
        probe = _first_day_next_month(y, m)
    return None


def _next_for_and_fast_path(term: list[dict], ref_d: date, seed: date) -> date:
    probe = ref_d
    stalled = 0
    for _ in range(MAX_ANCHOR_ITER):
        cands = [next_after_atom_with_mods(atom, probe, seed) for atom in term]
        if not cands:
            raise ParseError("Anchor evaluation term is empty; check anchor spec.")
        target = max(cands)
        if target <= probe:
            stalled += 1
            if stalled < 3:
                probe = probe + timedelta(days=1)
                continue
            if os.environ.get("NAUTICAL_DIAG") == "1":
                _warn_once_per_day(
                    "next_for_and_no_progress",
                    "[nautical] _next_for_and made no progress; failing fast. Check anchor spec.",
                )
            raise ParseError("Anchor evaluation made no forward progress; check anchor spec.")
        stalled = 0
        if all(atom_matches_on(atom, target, seed) for atom in term):
            return target
        probe = target
    if os.environ.get("NAUTICAL_DIAG") == "1":
        _warn_once_per_day(
            "next_for_and_fallback",
            f"[nautical] _next_for_and fallback after {MAX_ANCHOR_ITER} iterations.",
        )
    return ref_d + timedelta(days=365)


def _next_for_and(term: list[dict], ref_d: date, seed: date) -> date:
    """
    Find the next date > ref_d satisfying ALL atoms in 'term'.
    Rand-aware: if the term contains m:rand and any y:, choose the random
    day from the intersection of ALL constraints for each candidate month.
    Otherwise, fall back to the fast alignment loop.
    """
    has_m_rand = _term_has_monthly_rand(term)
    y_specs = _term_year_specs(term)
    if has_m_rand and y_specs:
        rand_yearly = _next_for_and_rand_yearly(term, ref_d, y_specs)
        if rand_yearly is not None:
            return rand_yearly
        return ref_d + timedelta(days=365)
    return _next_for_and_fast_path(term, ref_d, seed)


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
                          default_due_dt=None) -> AnchorHintsPayload:
    key = cache_key_for_task(anchor_expr, anchor_mode)
    cached = cache_load(key)
    if cached:
        return cast(AnchorHintsPayload, cached)

    dnf = validate_anchor_expr_strict(anchor_expr)
    natural = _describe_anchor_expr_from_dnf(dnf, default_due_dt=default_due_dt)
    hints = precompute_hints(dnf, start_dt=default_due_dt, anchor_mode=anchor_mode)

    payload = {
        "meta": {"created": int(time.time()),
                 "cfg": {"fmt": ANCHOR_YEAR_FMT, "salt": WRAND_SALT, "tz": LOCAL_TZ_NAME, "hol": HOLIDAY_REGION}},
        "dnf": dnf,
        "natural": natural,
        **hints,
    }
    cache_save(key, payload)
    return cast(AnchorHintsPayload, payload)


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
_Q_FIRST_MONTH_TOKEN_REV = {v: k for k, v in _Q_FIRST_MONTH_TOKEN.items()}
_Q_START_DAY_REV = {v: k for k, v in _Q_START_DAY.items()}
_Q_END_DAY_REV = {v: k for k, v in _Q_END_DAY.items()}


def _yearly_tokens(term):
    return _quarter_helpers.yearly_tokens(term, split_csv_tokens=_split_csv_tokens)


def _monthly_tokens(term):
    return _quarter_helpers.monthly_tokens(term, split_csv_tokens=_split_csv_tokens)


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
    return _quarter_helpers.quarters_from_tokens(y_toks, token_rev=_Q_FIRST_MONTH_TOKEN_REV)


def _quarters_from_start_day_tokens(y_toks):
    return _quarter_helpers.quarters_from_tokens(y_toks, token_rev=_Q_START_DAY_REV)


def _quarters_from_end_day_tokens(y_toks):
    return _quarter_helpers.quarters_from_tokens(y_toks, token_rev=_Q_END_DAY_REV)


def _format_quarter_set(qs):
    return _quarter_helpers.format_quarter_set(qs)


def _rewrite_quarter_spec_mode(spec: str, mode: str, meta_out: dict | None = None) -> str:
    return _quarter_rewrite.rewrite_quarter_spec_mode(
        spec,
        mode,
        meta_out=meta_out,
        split_csv_lower=_split_csv_lower,
        tok_range=_tok_range,
        static_month_last_day=_static_month_last_day,
        quarter_pos_month=_QUARTER_POS_MONTH,
        re_mod=re,
    )



_MONTH_SELECTOR_MAX_LEN = _quarter_selector.MONTH_SELECTOR_MAX_LEN


def _quarter_atom_spec(atom: dict) -> str:
    return _quarter_selector.quarter_atom_spec(atom)


def _has_quarter_tokens(spec: str) -> bool:
    return _quarter_selector.has_quarter_tokens(spec, split_csv_lower=_split_csv_lower, re_mod=re)


def _has_plain_quarter_tokens(spec: str) -> bool:
    return _quarter_selector.has_plain_quarter_tokens(spec, split_csv_lower=_split_csv_lower, re_mod=re)


def _is_negative_ascii_int(tok: str) -> bool:
    return _quarter_selector.is_negative_ascii_int(tok)


def _is_start_month_selector(tok: str) -> bool:
    return _quarter_selector.is_start_month_selector(
        tok,
        parse_error_cls=ParseError,
        safe_match=_safe_match,
        nth_weekday_re=_nth_weekday_re,
    )


def _is_end_month_selector(tok: str) -> bool:
    return _quarter_selector.is_end_month_selector(
        tok,
        parse_error_cls=ParseError,
        safe_match=_safe_match,
        nth_weekday_re=_nth_weekday_re,
        bd_re=_bd_re,
    )


def _quarter_month_selector_mode(m_atoms: list[dict]) -> str:
    return _quarter_selector.quarter_month_selector_mode(
        m_atoms,
        parse_error_cls=ParseError,
        expand_monthly_aliases=_expand_monthly_aliases,
        split_csv_tokens=_split_csv_tokens,
        is_start_month_selector=_is_start_month_selector,
        is_end_month_selector=_is_end_month_selector,
    )


def _term_quarter_rewrite_mode(y_atoms: list[dict], m_atoms: list[dict]) -> str:
    return _quarter_selector.term_quarter_rewrite_mode(
        y_atoms,
        m_atoms,
        quarter_atom_spec=_quarter_atom_spec,
        has_plain_quarter_tokens=_has_plain_quarter_tokens,
        quarter_month_selector_mode=_quarter_month_selector_mode,
    )


def _rewrite_quarter_year_atoms(y_atoms: list[dict], mode: str) -> None:
    _quarter_rewrite.rewrite_quarter_year_atoms(
        y_atoms,
        mode,
        quarter_atom_spec=_quarter_atom_spec,
        has_quarter_tokens=_has_quarter_tokens,
        rewrite_quarter_spec_mode=_rewrite_quarter_spec_mode,
    )


def _rewrite_quarters_in_context(dnf):
    return _quarter_rewrite.rewrite_quarters_in_context(
        dnf,
        has_quarter_tokens=_has_quarter_tokens,
        quarter_atom_spec=_quarter_atom_spec,
        term_quarter_rewrite_mode=_term_quarter_rewrite_mode,
        rewrite_quarter_year_atoms=_rewrite_quarter_year_atoms,
    )


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




def _describe_monthly_tokens(spec: str):
    return _split_csv_lower(spec)


def _describe_is_pure_nth_weekday_spec(spec: str):
    toks = _describe_monthly_tokens(spec)
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


def _describe_is_pure_dom_spec(spec: str):
    toks = _describe_monthly_tokens(spec)
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


def _describe_single_full_month_from_yearly_spec(spec: str):
    m = _year_range_colon_re.match(str(spec or "").strip())
    if not m:
        return None
    d1, m1, d2, m2 = map(int, m.groups())
    if m1 != m2 or d1 != 1:
        return None
    if d2 < 28 or d2 > 31:
        return None
    return m1


def _describe_term_roll_shift(term) -> str | None:
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


def _describe_term_bd_filter(term) -> bool:
    return any((a.get("mods") or {}).get("bd") for a in term)


def _describe_roll_suffix(roll: str) -> str:
    if roll == "pbd":
        return " if business day; otherwise the previous business day"
    if roll == "nbd":
        return " if business day; otherwise the next business day"
    if roll == "nw":
        return " if business day; otherwise the nearest business day (Fri if Saturday, Mon if Sunday)"
    return ""


def _describe_inject_schedule_suffixes(txt: str, term) -> str:
    roll = _describe_term_roll_shift(term)
    if roll:
        suffix = _describe_roll_suffix(roll)
    elif _describe_term_bd_filter(term):
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
        head, _sep, tail = txt.partition(" at ")
        return f"{head}{suffix} at {tail}"
    return txt + suffix


def _describe_anchor_term_collect(term):
    m_parts = []
    y_parts = []
    w_phrase = None
    bd_filter = False
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
                w_phrase = f"one random day every {wk_ival} weeks"
        elif typ == "m":
            mo_ival = max(mo_ival, ival)
            monthly_specs.append(spec)
            for tok in _split_csv_tokens(spec):
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

    return w_phrase, m_parts, y_parts, bd_filter, wk_ival, mo_ival, yr_ival, monthly_specs, yearly_specs


def _describe_anchor_term_fused_month_year(
    term,
    default_due_dt,
    monthly_specs,
    yearly_specs,
    yr_ival: int,
    bd_filter: bool,
    m_parts: list[str],
) -> str | None:
    if len(monthly_specs) != 1 or len(yearly_specs) != 1:
        return None
    mspec = monthly_specs[0]
    yspec = yearly_specs[0]
    is_nth, pairs = _describe_is_pure_nth_weekday_spec(mspec)
    fuse_month = _describe_single_full_month_from_yearly_spec(yspec)
    if not (is_nth and fuse_month and len(pairs) == 1):
        return None
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
    return _describe_inject_schedule_suffixes(main, term)


def _describe_anchor_term_interval_prefix(wk_ival, mo_ival, yr_ival, monthly_specs):
    interval_prefix = None
    suppress_tail = False

    if wk_ival > 1:
        interval_prefix = f"every {wk_ival} weeks: "
    elif mo_ival > 1:
        monthly_prefix = f"every {mo_ival} months"
        clarifier = ""
        if len(monthly_specs) == 1:
            mspec = monthly_specs[0]
            is_nth, pairs = _describe_is_pure_nth_weekday_spec(mspec)
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
                is_dom, doms = _describe_is_pure_dom_spec(mspec)
                if is_dom and any(d >= 29 for d in doms):
                    clarifier = (
                        f" among months that have day {doms[0]}"
                        if len(doms) == 1
                        else " among months that have those days"
                    )

        if clarifier:
            interval_prefix = monthly_prefix + clarifier
            suppress_tail = True
        else:
            interval_prefix = monthly_prefix + ": "
    elif yr_ival > 1:
        interval_prefix = f"every {yr_ival} years: "

    return interval_prefix, suppress_tail


def _describe_anchor_term_parts(w_phrase, m_parts, y_parts, bd_filter: bool) -> list[str]:
    parts = []
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
        if yp.startswith("one random day"):
            parts.append(yp)
        elif w_phrase or m_parts:
            parts.append(f"and within {yp}")
        else:
            parts.append(yp)

    if bd_filter and any("random day each month" in p for p in m_parts):
        parts.append("on a business day")
    return parts


def describe_anchor_term(term: list, default_due_dt=None) -> str:
    (
        w_phrase,
        m_parts,
        y_parts,
        bd_filter,
        wk_ival,
        mo_ival,
        yr_ival,
        monthly_specs,
        yearly_specs,
    ) = _describe_anchor_term_collect(term)

    fused = _describe_anchor_term_fused_month_year(
        term, default_due_dt, monthly_specs, yearly_specs, yr_ival, bd_filter, m_parts
    )
    if fused is not None:
        return fused

    interval_prefix, suppress_tail = _describe_anchor_term_interval_prefix(
        wk_ival, mo_ival, yr_ival, monthly_specs
    )
    parts = _describe_anchor_term_parts(w_phrase, m_parts, y_parts, bd_filter)
    hhmm = _fmt_hhmm_for_term(term, default_due_dt)

    if suppress_tail:
        txt = interval_prefix
        if hhmm:
            txt = f"{txt} at {hhmm}"
        return _describe_inject_schedule_suffixes(txt or "any day", term)

    if hhmm:
        parts.append(f"at {hhmm}")
    txt = " ".join(p for p in parts if p)
    if interval_prefix:
        txt = interval_prefix + txt

    txt = _inject_prevnext_phrase(txt, term)
    txt = _describe_inject_schedule_suffixes(txt or "any day", term)
    return txt or "any day"

def _describe_anchor_expr_from_dnf(dnf: list, default_due_dt=None) -> str:
    nat_terms = []
    for term in (dnf or []):
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
    return ordered[0] if len(ordered) == 1 else " or ".join(ordered)


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
    return _describe_anchor_expr_from_dnf(dnf, default_due_dt=default_due_dt)



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


def _join_natural_or_terms(terms: list[str]) -> str:
    if not terms:
        return ""
    if len(terms) == 1:
        return terms[0]
    if len(terms) == 2:
        return f"either {terms[0]} or {terms[1]}"
    return "either " + ", ".join(terms[:-1]) + ", or " + terms[-1]


def _longest_common_suffix(parts: list[str]) -> str:
    if not parts:
        return ""
    rev = [p[::-1] for p in parts if isinstance(p, str)]
    if not rev:
        return ""
    prefix = os.path.commonprefix(rev)
    return prefix[::-1]


def _compress_or_terms_by_clause(terms: list[str], delim: str) -> str | None:
    """Compact repeated OR terms of the form '<shared><delim><variant><shared-tail>'."""
    if not terms or len(terms) < 2:
        return None

    if not isinstance(delim, str) or not delim:
        return None
    split: list[tuple[str, str]] = []
    for t in terms:
        if not isinstance(t, str):
            return None
        idx = t.find(delim)
        if idx <= 0:
            return None
        prefix = t[:idx]
        rest = t[idx + len(delim):]
        if not rest:
            return None
        split.append((prefix, rest))

    prefixes = {p for p, _ in split}
    if len(prefixes) != 1:
        return None
    prefix = split[0][0]
    rests = [r for _, r in split]
    if len(set(rests)) <= 1:
        return None

    common_tail = _longest_common_suffix(rests)
    if delim == " and within ":
        # Avoid over-compressing calendar-day yearly phrases into "Jan, Feb, ... Oct 1 each year".
        # Prefer keeping day with each variant: "Jan 1, Feb 1, ... Oct 1 each year".
        if (
            common_tail
            and re.match(r"^\s+\d{1,2}\b", common_tail)
            and "each year" in common_tail
        ):
            alt_tail = " each year"
            if all(r.endswith(alt_tail) for r in rests):
                common_tail = alt_tail
    variants: list[str] = []
    for r in rests:
        v = r[:-len(common_tail)] if common_tail else r
        v = v.strip(" ,")
        if not v:
            return None
        variants.append(v)

    joined = _join_natural_or_terms(variants)
    return f"{prefix}{delim}{joined}{common_tail}"


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
    # Deduplicate while preserving order before joining/compression.
    seen = set()
    uniq_terms = []
    for t in terms:
        if t and t not in seen:
            seen.add(t)
            uniq_terms.append(t)
    if not uniq_terms:
        return ""
    sentence = (
        _compress_or_terms_by_clause(uniq_terms, " and within ")
        or _compress_or_terms_by_clause(uniq_terms, " that fall on ")
        or _join_natural_or_terms(uniq_terms)
    )
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


def _rand_bucket_time_from_mods(mods: dict) -> str | None:
    tmod = mods.get("t")
    if isinstance(tmod, tuple):
        return f"{tmod[0]:02d}:{tmod[1]:02d}"
    if isinstance(tmod, str) and tmod:
        return tmod
    return None


def _rand_bucket_merge_mods(mods: dict, time_str: str | None, bd_flag: bool) -> tuple[str | None, bool]:
    if time_str is None:
        time_str = _rand_bucket_time_from_mods(mods)
    bd_flag = bd_flag or bool(mods.get("bd") or (mods.get("wd") is True))
    return time_str, bd_flag


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

    for a in term:
        typ = (a.get("typ") or a.get("type") or "").lower()
        if typ in ("w", "y") or typ != "m":
            return None
        spec = str(a.get("spec") or a.get("value") or "").lower()
        ival = int(a.get("ival") or a.get("intv") or 1)
        ival_seen = ival if ival_seen is None else ival_seen
        mods = a.get("mods") or {}
        time_str, bd_flag = _rand_bucket_merge_mods(mods, time_str, bd_flag)
        if spec == "rand":
            has_rand = True
            continue
        rn = _normalize_range_token(spec)
        if not rn:
            return None
        if range_norm and rn != range_norm:
            return None
        range_norm = rn

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

    if sig is None:
        return None
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
    out: list[str] = []
    buf: list[str] = []
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
    return _common.coerce_int(v, default=default)


def parse_dt_any(s: str):
    return _timeutil.parse_dt_any(s, DATE_FORMATS)

from . import dates as _dates


def month_len(y, m):
    return _dates.month_len(y, m)


def add_months(d: date, months: int) -> date:
    return _dates.add_months(d, months)


def months_days_between(d1: date, d2: date):
    return _dates.months_days_between(d1, d2)


def humanize_delta(from_dt: datetime, to_dt: datetime, use_months_days: bool):
    return _dates.humanize_delta(from_dt, to_dt, use_months_days)


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
    out: set[int] = set()

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
def _normalize_anchor_expr_input(s: str) -> str:
    """Normalize user anchor expression before parsing."""
    s = _unwrap_quotes(s or "").strip()
    if len(s) > 1024:
        raise ParseError("Anchor expression too long (max 1024 characters).")
    s = re.sub(r"\b(\d{2})-rand\b", r"rand-\1", s)
    s = _rewrite_weekly_multi_time_atoms(s)
    return s


def _fatal_bad_colon_in_year_tail(tail: str) -> str | None:
    head = tail.split("@", 1)[0]
    for tok in _split_csv_tokens(head):
        if re.fullmatch(r"\d{2}:\d{2}(?::\d{2}:\d{2})?", tok):
            fmt = (globals().get("ANCHOR_YEAR_FMT") or "DM").upper()
            example = "06-01" if fmt == "MD" else "01-06"
            return (
                f"Yearly token '{tok}' uses ':' between numbers. "
                f"Use '-' and order per ANCHOR_YEAR_FMT={fmt}. Example: '{example}'."
            )
        if ":" in tok:
            return "Yearly ranges must use '..' (e.g., '01-01..12-31', 'q1..q2')."
    return None


def _raise_on_bad_colon_year_tokens(s: str) -> None:
    # Find every 'y' atom head (with optional /N) then validate its tail.
    for m in re.finditer(r"\by\s*(?:/\d+)?\s*:", s):
        j = m.end()
        k = j
        while k < len(s) and s[k] not in "+|)":
            k += 1
        tail = s[j:k]
        fatal_msg = _fatal_bad_colon_in_year_tail(tail)
        if fatal_msg:
            raise ParseError(fatal_msg)


def _skip_ws_pos(s: str, i: int, n: int) -> int:
    while i < n and s[i].isspace():
        i += 1
    return i


def _raise_if_comma_joined_anchors(full_tail: str) -> None:
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


def _normalize_monthly_ordinal_spec(spec: str) -> str:
    def _ord_norm(mo: re.Match) -> str:
        return f"{mo.group(1)}{mo.group(3).lower()}"

    return re.sub(
        r"\b([1-5])\s*(st|nd|rd|th)\s*-\s*(mon|tue|wed|thu|fri|sat|sun)\b",
        _ord_norm,
        spec,
        flags=re.IGNORECASE,
    )


def _build_anchor_atom_dnf(head: str, full_tail: str):
    """Build DNF node for one parsed atom head/tail pair."""
    typ, ival = _parse_atom_head(head)
    tlo = (typ or "").lower()

    dnf_or = _parse_group_with_inline_mods(tlo, ival, full_tail, "")
    if dnf_or is not None:
        return dnf_or

    spec, mods_str = (full_tail.split("@", 1) + [""])[:2]
    if tlo == "m":
        spec = _normalize_monthly_ordinal_spec(spec)

    if tlo == "w":
        toks = _split_csv_lower(spec)
        if "rand" in toks and len(toks) > 1:
            mods = _parse_atom_mods(mods_str)
            return [
                [{"typ": "w", "spec": t, "ival": ival, "mods": mods}]
                for t in toks
            ]

    mods = _parse_atom_mods(mods_str)
    return [[{"typ": tlo, "spec": spec.strip().lower(), "ival": ival, "mods": mods}]]


def _parse_anchor_atom_at(s: str, i: int, n: int):
    """Parse one atom at position i and return (dnf_node, next_i)."""
    i = _skip_ws_pos(s, i, n)

    start = i
    while i < n and s[i] not in ":()+|":
        i += 1
    head = s[start:i].strip()

    if i >= n or s[i] != ":":
        raise ParseError("Expected ':' after anchor type. Example 'w:mon', 'm:-1', 'y:06-01'")
    i += 1

    start = i
    while i < n:
        ch = s[i]
        if ch in ")|":
            break
        if ch == "+" and not (i > start and s[i - 1] == "@"):
            break
        i += 1

    full_tail = s[start:i].strip()
    _raise_if_comma_joined_anchors(full_tail)
    return _build_anchor_atom_dnf(head, full_tail), i


def parse_anchor_expr_to_dnf(s: str) -> AnchorDNF:
    """Parse anchor expression into Disjunctive Normal Form."""
    s = _normalize_anchor_expr_input(s)
    _raise_on_bad_colon_year_tokens(s)

    i = 0
    n = len(s)

    def parse_atom():
        nonlocal i
        node, i = _parse_anchor_atom_at(s, i, n)
        return node

    def parse_factor(depth: int = 0):
        nonlocal i
        if depth > 50:
            raise ParseError("Expression nesting too deep")
        i = _skip_ws_pos(s, i, n)
        if i < n and s[i] == "(":
            i += 1
            res = parse_expr(depth + 1)
            i = _skip_ws_pos(s, i, n)
            if i >= n or s[i] != ")":
                raise ParseError("Unclosed '('")
            i += 1
            return res
        return parse_atom()

    def and_merge(A, B):
        out: list[list[dict]] = []
        for ta in A:
            for tb in B:
                out.append(ta + tb)
                if len(out) > MAX_ANCHOR_DNF_TERMS:
                    raise ParseError(
                        f"Expression too complex: more than {MAX_ANCHOR_DNF_TERMS} combined terms."
                    )
        return out

    def parse_term(depth: int = 0):
        nonlocal i
        left = parse_factor(depth)
        while True:
            pos = i
            i = _skip_ws_pos(s, i, n)
            if i >= n or s[i] != "+":
                i = pos
                break
            i += 1
            right = parse_factor(depth)
            left = and_merge(left, right)
        return left

    def parse_expr(depth: int = 0):
        nonlocal i
        left = parse_term(depth)
        while True:
            pos = i
            i = _skip_ws_pos(s, i, n)
            if i >= n or s[i] != "|":
                i = pos
                break
            i += 1
            right = parse_term(depth)
            if len(left) + len(right) > MAX_ANCHOR_DNF_TERMS:
                raise ParseError(
                    f"Expression too complex: more than {MAX_ANCHOR_DNF_TERMS} OR terms."
                )
            left = left + right
        return left

    res = parse_expr(0)
    i = _skip_ws_pos(s, i, n)
    if i != n:
        raise ParseError("Unexpected trailing characters")
    dnf = _rewrite_quarters_in_context(res)
    dnf = _rewrite_year_month_aliases_in_context(dnf)   # NEW
    _validate_year_tokens_in_dnf(dnf)
    _validate_and_terms_satisfiable(dnf, ref_d=date.today())
    return dnf


@_ttl_lru_cache(maxsize=256)
def _parse_anchor_expr_to_dnf_cached_obj(s: str, fmt: str) -> AnchorDNF:
    return parse_anchor_expr_to_dnf(s)


def parse_anchor_expr_to_dnf_cached(s: str) -> AnchorDNF:
    """Cached parse returning a fresh object (avoid shared mutable structures)."""
    if not s:
        return []
    key = _unwrap_quotes(s or "").strip()
    if not key:
        return []
    res = _clone_dnf(_parse_anchor_expr_to_dnf_cached_obj(key, _yearfmt()))
    _emit_cache_metrics()
    if os.environ.get("NAUTICAL_CLEAR_CACHES") == "1":
        _clear_all_caches()
    return res



# ------------------------------------------------------------------------------
# Anchor validators
# ------------------------------------------------------------------------------
class YearTokenFormatError(ParseError):
    pass


def _yearly_pair_from_fmt(a: int, b: int, fmt: str) -> tuple[int, int]:
    # returns (day, month)
    return (b, a) if fmt == "MD" else (a, b)


def _yearly_mmdd_error(mm: int, dd: int) -> str | None:
    if not (1 <= mm <= 12):
        return f"month '{mm:02d}' is invalid"
    if not (1 <= dd <= 31):
        return f"day '{dd:02d}' is invalid"
    return None


def _validate_yearly_token_allowlist(tok: str, fmt: str) -> None:
    s = tok

    # Allow 'rand' and 'rand-XX'
    if s == "rand" or re.fullmatch(r"rand-\d{2}", s):
        return

    # FATAL: numeric with ':' (e.g., '05:15' or '05:01:06:30')
    if re.fullmatch(r"\d{2}:\d{2}(?::\d{2}:\d{2})?", s):
        example = "06-01" if fmt == "MD" else "01-06"
        raise YearTokenFormatError(
            f"Yearly token '{tok}' uses ':' between numbers. "
            f"Use '-' and order per ANCHOR_YEAR_FMT={fmt}. Example: '{example}'."
        )

    # Accept standard numeric day-month (single or range) — parser ensures order (MD/DM)
    if re.fullmatch(r"\d{2}-\d{2}(?:\.\.\d{2}-\d{2})?", s):
        return

    # Accept month aliases and month..month; rewritten downstream
    if re.fullmatch(r"(?:[a-z]{3}|\d{2})\.\.(?:[a-z]{3}|\d{2})", s):
        return
    if re.fullmatch(r"[a-z]{3}", s):  # 'apr', 'jul'
        return

    # Accept quarters (rewritten earlier)
    if re.fullmatch(r"q[1-4](?:\.\.q[1-4])?", s):
        return

    # Everything else is invalid
    raise YearTokenFormatError(f"Unknown yearly token '{tok}'. Expected day-month, month alias, or quarter.")


def _validate_yearly_token_detailed(tok: str, fmt: str) -> tuple[str, str] | None:
    s = tok.strip().lower()

    if s == "rand":
        return None

    m_randm = re.fullmatch(r"rand-(\d{2})", s)
    if m_randm:
        mm = int(m_randm.group(1))
        if 1 <= mm <= 12:
            return None
        raise YearTokenFormatError(f"Invalid month in yearly token '{tok}'. Expected 01..12.")

    # Proper numeric tokens: DD-MM or MM-DD, with optional range tail (V2 '..')
    m = re.fullmatch(r"(\d{2})-(\d{2})(?:\.\.(\d{2})-(\d{2}))?$", s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        d1, m1 = _yearly_pair_from_fmt(a, b, fmt)
        err = _yearly_mmdd_error(m1, d1)
        if err:
            return tok, err
        if m.group(3):
            c, d = int(m.group(3)), int(m.group(4))
            d2, m2 = _yearly_pair_from_fmt(c, d, fmt)
            err2 = _yearly_mmdd_error(m2, d2)
            if err2:
                return tok, err2
            if (m2, d2) < (m1, d1):
                return tok, "end precedes start"
        return None

    # colon-only separators like '05:15' or '05:15:06:20' -> friendly error
    m_col1 = re.fullmatch(r"(\d{2}):(\d{2})$", s)
    m_col2 = re.fullmatch(r"(\d{2}):(\d{2}):(\d{2}):(\d{2})$", s)
    if m_col1 or m_col2:
        if m_col1:
            A, B = int(m_col1.group(1)), int(m_col1.group(2))
            ex = f"{A:02d}-{B:02d}" if fmt == "MD" else f"{B:02d}-{A:02d}"
        else:
            A, B, C, D = map(int, m_col2.groups())
            ex = (
                f"{A:02d}-{B:02d}..{C:02d}-{D:02d}"
                if fmt == "MD"
                else f"{B:02d}-{A:02d}..{D:02d}-{C:02d}"
            )
        raise YearTokenFormatError(
            f"Yearly token '{tok}' uses ':' between numbers. "
            f"Use '-' and order per ANCHOR_YEAR_FMT={fmt}. Example: '{ex}'."
        )

    if ":" in s:
        raise YearTokenFormatError(
            "Yearly ranges must use '..' (e.g., '01-01..12-31', 'q1..q2')."
        )

    # If it looks numeric-ish but didn’t match the proper pattern, nudge with a general hint.
    if any(ch.isdigit() for ch in s) and any(ch in s for ch in "-:"):
        ex = "MM-DD" if fmt == "MD" else "DD-MM"
        raise YearTokenFormatError(
            f"Yearly token '{tok}' doesn’t match ANCHOR_YEAR_FMT={fmt}. "
            f"Expected {ex} or {ex}..{ex}."
        )

    # Non-numeric tokens (e.g., month names/quarters) are rewritten earlier and can pass here.
    return None


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
        _validate_yearly_token_allowlist(tok, fmt)
    bad = None

    for tok in tokens:
        bad = _validate_yearly_token_detailed(tok, fmt)
        if bad:
            break

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


_YEARLY_MONTH_MAX = {
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
_YEARLY_QUARTER_RE = re.compile(r"^q[1-4][sme]?$")
_YEARLY_QUARTER_RANGE_RE = re.compile(r"^(q[1-4])([sme])?\.\.(q[1-4])([sme])?$")
_YEARLY_MONTH_ONLY_RE = re.compile(r"^\d{1,2}$")  # '3' or '03'
_YEARLY_MONTH_RANGE_ONLY_RE = re.compile(r"^\d{1,2}\.\.\d{1,2}$")  # '3..4'
_YEARLY_NON_PADDED_DM_RE = re.compile(r"^\d{1,2}-\d{1,2}(?:\.\.\d{1,2}-\d{1,2})?$")
_YEARLY_PADDED_DM_RE = re.compile(
    r"^(?P<d1>\d{2})-(?P<m1>\d{2})(?:\.\.(?P<d2>\d{2})-(?P<m2>\d{2}))?$"
)


def _yearly_last_day(mm: int) -> int:
    return _YEARLY_MONTH_MAX.get(mm, 31)


def _yearly_check_day_month(dd: int, mm: int, label: str, tok: str) -> None:
    if mm < 1 or mm > 12:
        raise ParseError(
            f"Invalid month '{mm:02d}' in '{tok}' ({label}). Months must be 01..12."
        )
    maxd = _yearly_last_day(mm)
    if dd < 1 or dd > maxd:
        near = maxd if dd > maxd else 1
        hint = f" {_MONTH_FULL[mm]} has {maxd} days."
        sug1 = f"{near:02d}-{mm:02d}"
        sug2 = f"01-{mm:02d}..{maxd:02d}-{mm:02d}"
        raise ParseError(
            f"Invalid day '{dd:02d}' for month '{mm:02d}' in '{tok}' ({label}).{hint} "
            f"Try '{sug1}' or '{sug2}'."
        )


def _validate_yearly_spec_token(tok: str) -> None:
    if ":" in tok:
        raise ParseError(
            "Yearly ranges must use '..' (e.g., '01-01..12-31', 'q1..q2')."
        )
    # --- Quarters (single) ---
    if _YEARLY_QUARTER_RE.fullmatch(tok):
        return

    # --- Quarter ranges like 'q1..q2' (monotonic only) ---
    m = _YEARLY_QUARTER_RANGE_RE.fullmatch(tok)
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
        return

    # --- Month-only like '03' → suggest full month ---
    if _YEARLY_MONTH_ONLY_RE.match(tok):
        mm = int(tok)
        if not (1 <= mm <= 12):
            raise ParseError(
                f"Invalid month '{tok}'. Months must be 01..12. "
                f"Try '{mm:02d}' or the full-month form '01-{mm:02d}..{_yearly_last_day(mm):02d}-{mm:02d}'."
            )
        raise ParseError(
            f"Yearly token '{tok}' is incomplete. Did you mean the full month? "
            f"Try '01-{mm:02d}..{_yearly_last_day(mm):02d}-{mm:02d}'."
        )

    # --- 'MM..MM' → suggest full multi-month range with proper end day ---
    if _YEARLY_MONTH_RANGE_ONLY_RE.match(tok):
        m1, m2 = (int(x) for x in tok.split("..", 1))
        if not (1 <= m1 <= 12 and 1 <= m2 <= 12):
            raise ParseError(
                f"Invalid month range '{tok}'. Months must be 01..12. "
                f"Try '01-{m1:02d}..{_yearly_last_day(m2):02d}-{m2:02d}'."
            )
        if m2 < m1:
            left = f"01-{m1:02d}..31-12"
            right = f"01-01..{_yearly_last_day(m2):02d}-{m2:02d}"
            raise ParseError(
                f"Invalid month range '{tok}': end month is before start month. "
                f"Split across years, e.g., '{left}, {right}'."
            )
        raise ParseError(
            f"Yearly token '{tok}' is incomplete. Did you mean a full multi-month range? "
            f"Try '01-{m1:02d}..{_yearly_last_day(m2):02d}-{m2:02d}'."
        )

    # --- Zero-padding guidance for DM or DM..DM ---
    if _YEARLY_NON_PADDED_DM_RE.match(tok) and not _YEARLY_PADDED_DM_RE.match(tok):
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
    m = _YEARLY_PADDED_DM_RE.match(tok)
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
        _yearly_check_day_month(d1, m1, "single day", tok)
        return

    d2 = int(d2g)
    m2 = int(m2g)
    _yearly_check_day_month(d1, m1, "range start", tok)
    _yearly_check_day_month(d2, m2, "range end", tok)
    if (m2, d2) < (m1, d1):
        left = f"{d1:02d}-{m1:02d}..31-12"
        right = f"01-01..{d2:02d}-{m2:02d}"
        raise ParseError(
            f"Invalid range '{tok}': start must be on/before end; cross-year ranges "
            f"aren't supported. Try splitting: '{left}, {right}'."
        )


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
    for tok in toks:
        _validate_yearly_spec_token(tok)


def _normalize_anchor_input_to_dnf(expr) -> AnchorDNF:
    """Normalize user input to parsed DNF, preserving current error messages."""
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

    # Defensive compatibility for legacy tuple-style parser errors.
    if isinstance(dnf, tuple) and len(dnf) == 2 and isinstance(dnf[0], str):
        raise ParseError(dnf[0])
    return dnf


def _assert_dnf_structure_strict(dnf):
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


def _validate_anchor_atom_strict(a: dict) -> None:
    typ = (a.get("typ") or a.get("type") or "").lower()
    spec = (a.get("spec") or a.get("value") or "").lower()
    ival = int(a.get("ival") or a.get("intv") or 1)
    mods = a.get("mods") or {}
    active = None

    if typ == "w":
        _validate_weekly_spec(spec)
        return

    if typ == "m":
        if spec == "rand":
            active = _active_mod_keys(mods)
            bad = [k for k in active if k not in ("t", "bd", "wd")]
            if bad:
                raise ParseError(f"m:rand does not support @{', '.join(bad)}")
            if ival < 1:
                raise ParseError("Monthly interval (/N) must be >= 1")
        else:
            _validate_monthly_spec(spec)
        return

    if typ == "y":
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
        return

    raise ParseError(f"Unknown anchor type '{typ}'")


def _validate_anchor_dnf_atoms_strict(dnf: AnchorDNF) -> None:
    for term in dnf:
        for a in term:
            _validate_anchor_atom_strict(a)


def validate_anchor_expr_strict(expr) -> AnchorDNF:
    """
    Validate an anchor expression. Accepts:
      - str  (e.g., "w/2:sun + m:1st-mon"), parsed to DNF
      - DNF  (list[list[dict]]), already parsed

    Returns the normalized DNF on success; raises ParseError on failure.
    """
    dnf = _normalize_anchor_input_to_dnf(expr)
    _assert_dnf_structure_strict(dnf)
    _validate_anchor_dnf_atoms_strict(dnf)
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

    # Unknown type should not happen after strict validation; keep deterministic fallback.
    return ref_d + timedelta(days=365)


# ------------------------------------------------------------------------------
# Anchor scheduling & iteration helpers
# ------------------------------------------------------------------------------
def _interval_allowed_for_atom(typ: str, ival: int, seed: date, cand: date) -> bool:
    if ival <= 1:
        return True
    if typ == "w":
        weeks_diff = _weeks_between(seed, cand)
        return weeks_diff % ival == 0
    if typ == "y":
        return (_year_index(cand) - _year_index(seed)) % ival == 0
    return True


def _advance_probe_for_interval_bucket(typ: str, ival: int, seed: date, cand: date) -> date:
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


def _month_doms_safe(spec: str, y: int, m: int) -> list[int]:
    try:
        return sorted(expand_monthly_cached(spec, y, m))
    except Exception:
        return []


def _month_has_hit(spec: str, y: int, m: int) -> bool:
    return bool(_month_doms_safe(spec, y, m))


def _first_hit_after_probe_in_month(spec: str, y: int, m: int, probe: date) -> date | None:
    for d0 in _month_doms_safe(spec, y, m):
        dt = date(y, m, d0)
        if dt > probe:
            return dt
    return None


def _next_valid_month_on_or_after(spec: str, y: int, m: int) -> tuple[int, int]:
    yy, mm = y, m
    for _ in range(480):
        if _month_has_hit(spec, yy, mm):
            return yy, mm
        mm += 1
        if mm > 12:
            yy += 1
            mm = 1
    return y, m


def _advance_k_valid_months(spec: str, start_y: int, start_m: int, k: int) -> tuple[int, int]:
    yy, mm = start_y, start_m
    steps = max(k, 0)
    while steps >= 0:
        mm += 1
        if mm > 12:
            mm = 1
            yy += 1
        yy, mm = _next_valid_month_on_or_after(spec, yy, mm)
        steps -= 1
    return yy, mm


def _monthly_align_base_for_interval(spec: str, base: date, probe: date, seed: date, ival: int) -> date:
    by, bm = base.year, base.month

    # Seed bucket = first valid month on/after seed.
    sy, sm = _next_valid_month_on_or_after(spec, seed.year, seed.month)

    # Ensure base is in a valid month and strictly > probe.
    if not _month_has_hit(spec, by, bm):
        by, bm = _next_valid_month_on_or_after(spec, by, bm)
        nxt = _first_hit_after_probe_in_month(spec, by, bm, probe)
        if nxt is None:
            ny, nm = _advance_k_valid_months(spec, by, bm, 0)
            doms = _month_doms_safe(spec, ny, nm)
            base = date(ny, nm, doms[0])
        else:
            base = nxt
    elif base <= probe:
        nxt = _first_hit_after_probe_in_month(spec, by, bm, probe)
        if nxt is None:
            ny, nm = _advance_k_valid_months(spec, by, bm, 0)
            doms = _month_doms_safe(spec, ny, nm)
            base = date(ny, nm, doms[0])
        else:
            base = nxt

    # Count valid-month steps from (sy, sm) to base(y, m).
    cnt = 0
    ty, tm = sy, sm
    while (ty, tm) != (base.year, base.month) and cnt < 480:
        ty, tm = _advance_k_valid_months(spec, ty, tm, 0)
        cnt += 1

    if (cnt % ival) != 0:
        steps = ival - (cnt % ival)
        ny, nm = _advance_k_valid_months(spec, base.year, base.month, steps - 1)
        doms = _month_doms_safe(spec, ny, nm)
        base = date(ny, nm, doms[0])

    return base


def _accept_roll_candidate(ref_d: date, base: date, cand: date, roll_kind: str | None) -> bool:
    if roll_kind in ("pbd", "nbd", "nw"):
        # For business-day rolls, rolled candidate may land on ref_d.
        return base > ref_d and cand >= ref_d
    return cand > ref_d


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
    typ = atom["typ"]
    mods = atom.get("mods") or {}
    spec = atom.get("spec") or ""

    # Fast path: ival==1 and no modifiers
    if ival == 1 and not _active_mod_keys(mods):
        candidate = base_next_after_atom(atom, ref_d)
        if candidate > ref_d:
            return candidate

    # ---- guarded iteration ----
    for _ in range(MAX_ANCHOR_ITER):
        base = base_next_after_atom(atom, probe)

        # @bd modifier: weekdays only - skip weekends entirely (move to next bucket)
        if mods.get("bd") and base.weekday() > 4:  # 5=Saturday, 6=Sunday
            probe = base + timedelta(days=1)
            continue

        # weekly/yearly gating first
        if typ in ("w", "y") and not _interval_allowed_for_atom(typ, ival, seed, base):
            probe = _advance_probe_for_interval_bucket(typ, ival, seed, base)
            continue

        # monthly special-case: /N by *valid months*
        if typ == "m" and ival > 1:
            base = _monthly_align_base_for_interval(spec, base, probe, seed, ival)

        # --- apply roll + offsets and decide acceptance ---
        rolled = roll_apply(base, mods)
        cand = apply_day_offset(rolled, mods)

        roll_kind = mods.get("roll")
        if _accept_roll_candidate(ref_d, base, cand, roll_kind):
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


def _pick_earlier_candidate(
    best: date | None,
    best_meta: dict | None,
    cand: date | None,
    meta: dict | None,
):
    if cand and (best is None or cand < best):
        return cand, meta
    return best, best_meta


def _next_after_expr_monthly_rand_candidate(
    term: list[dict],
    term_id: int,
    info: dict,
    after_date: date,
    default_seed: date | None,
    seed_base: str | None,
):
    if any(_atype(a) == "y" for a in term):
        cand = _next_for_and(term, after_date, default_seed)
        if cand:
            return cand, {"basis": "rand+yearly"}
        return None, None

    seed_key_base = seed_base if seed_base is not None else "preview"
    mods = info.get("mods") or {}
    bd_only = bool(mods.get("bd"))
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
            seed_key = f"{seed_key_base}|m|{term_id}|{period_key}"
            idx = _sha_pick(len(cands), seed_key)
            choice = cands[idx]
            if choice > after_date:
                return choice, {"basis": "rand", "rand_period": period_key}
        m = 1 if m == 12 else m + 1
        if m == 1:
            y += 1

    return None, None


def _next_after_expr_yearly_rand_candidate(
    term: list[dict],
    term_id: int,
    info: dict,
    after_date: date,
    seed_base: str | None,
):
    seed_key_base = seed_base if seed_base is not None else "preview"
    mods = info.get("mods") or {}
    bd_only = bool(mods.get("bd"))
    target_m = info.get("month", None)
    y = after_date.year

    for _ in range(10):  # up to 10 years ahead
        if target_m is None:
            # plain y:rand -> gather all monthly candidates for this year
            cands = []
            for mm in range(1, 13):
                cands.extend(_term_candidates_in_month(term, y, mm, info["atom_idx"], bd_only))
            period_key = f"{y:04d}"
        else:
            cands = _term_candidates_in_month(term, y, int(target_m), info["atom_idx"], bd_only)
            period_key = f"{y:04d}-{int(target_m):02d}"

        if cands:
            seed_key = f"{seed_key_base}|y|{term_id}|{period_key}"
            idx = _sha_pick(len(cands), seed_key)
            choice = cands[idx]
            if choice > after_date:
                return choice, {"basis": "rand", "rand_period": period_key}
        y += 1

    return None, None


def _next_after_expr_term_candidate(term: list[dict], after_date: date, default_seed: date | None):
    cand, _ = next_after_term(term, after_date, default_seed)
    if cand:
        return cand, {"basis": "term"}
    return None, None


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
            cand, meta = _next_after_expr_monthly_rand_candidate(
                term, term_id, info, after_date, default_seed, seed_base
            )
            best, best_meta = _pick_earlier_candidate(best, best_meta, cand, meta)
            continue

        if rk == "y":
            cand, meta = _next_after_expr_yearly_rand_candidate(
                term, term_id, info, after_date, seed_base
            )
            best, best_meta = _pick_earlier_candidate(best, best_meta, cand, meta)
            continue

        cand, meta = _next_after_expr_term_candidate(term, after_date, default_seed)
        best, best_meta = _pick_earlier_candidate(best, best_meta, cand, meta)

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
    while len(acc) < UNTIL_COUNT_CAP:
        d, _ = next_after_expr(dnf, cur, default_seed, seed_base=seed_base)
        if d is None or d >= end_excl:
            break
        if d <= cur:
            if os.environ.get("NAUTICAL_DIAG") == "1":
                _warn_once_per_day(
                    "anchors_between_no_progress",
                    "[nautical] anchors_between_expr made no progress; stopping early.",
                )
            break
        if acc and d <= acc[-1]:
            cur = acc[-1] + timedelta(days=1)
            continue
        acc.append(d)
        cur = d + timedelta(days=1)
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
    return _timeutil.build_local_datetime(d, hhmm, _LOCAL_TZ)



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


def _lint_expand_year_month_aliases(s: str) -> str:
    # Allow bare month aliases: replace 'y:jun' with a canonical monthly window for linting.
    def _lint_month_alias_sub(m):
        mm = _month_from_alias(m.group(1))
        if not mm:
            return m.group(0)
        return f"y:{_year_full_month_range_token(mm)}"

    # Allow bare month aliases ONLY when they are not part of a numeric day-month like 'y:01-13'.
    #  - 'y:jan' or 'y:03' -> expand to full month window
    #  - do NOT touch 'y:01-13' / 'y:jun-01' etc.
    s = re.sub(r"\by:([a-z]{3})(?=\b(?!-)|[,+|()])", _lint_month_alias_sub, s)
    s = re.sub(r"\by:(\d{2})(?=(?:\b(?!-)|[,+|()]))", _lint_month_alias_sub, s)
    return s


def _lint_check_weekly_delimiter_contract(s: str) -> str | None:
    if re.search(r"\bw(?:/\d+)?\s*:\s*(?:mon|tue|wed|thu|fri|sat|sun)\s*-\s*(?:mon|tue|wed|thu|fri|sat|sun)\b", s):
        return "Weekly ranges must use '..' (e.g., 'w:mon..fri')."
    if re.search(r"\bw(?:/\d+)?\s*:\s*(?:mon|tue|wed|thu|fri|sat|sun)(?:\s*:\s*(?:mon|tue|wed|thu|fri|sat|sun))+\b", s):
        return "Weekly ranges must use '..' (e.g., 'w:mon..fri')."
    return None


def _lint_check_yearly_segments(s: str) -> str | None:
    fmt = _yearfmt()  # "MD" or "DM"
    for seg in _iter_y_segments(s):
        for tok in _split_csv_tokens(seg):
            if re.fullmatch(r"\d{2}:\d{2}", tok):
                return "Yearly day/month must use '-', not ':'. Try '05-15' (not '05:15')."
            if ":" in tok:
                return "Yearly ranges must use '..' (e.g., '01-01..12-31', 'q1..q2')."
            if re.fullmatch(r"\d{2}-\d{2}", tok):
                a, b = tok.split("-")
                x, y = int(a), int(b)
                if fmt == "MD":
                    if x > 12 and 1 <= y <= 12:
                        return f"'{tok}' looks like DD-MM but config expects MM-DD. Try '{y:02d}-{x:02d}'."
                else:  # DM
                    if y > 12 and 1 <= x <= 12:
                        return f"'{tok}' looks like MM-DD but config expects DD-MM. Try '{y:02d}-{x:02d}'."
                continue
            if re.fullmatch(r"\d{2}-\d{2}\.\.\d{2}-\d{2}", tok):
                left, right = tok.split("..", 1)
                a, b = left.split("-", 1)
                c, d = right.split("-", 1)
                x, y, u, v = int(a), int(b), int(c), int(d)
                if fmt == "MD":
                    if x > 12 and 1 <= y <= 12:
                        return (
                            f"'{tok}' starts like DD-MM but config expects MM-DD. "
                            f"Try '{y:02d}-{x:02d}..{v:02d}-{u:02d}'."
                        )
                else:
                    if y > 12 and 1 <= x <= 12:
                        return (
                            f"'{tok}' starts like MM-DD but config expects DD-MM. "
                            f"Try '{y:02d}-{x:02d}..{v:02d}-{u:02d}'."
                        )
                continue
    return None


def _lint_check_global_md_dm_confusion(s: str) -> str | None:
    for m in re.finditer(r"\b(\d{2})-(\d{2})(?=([^\d:]|$))", s):
        a, b = int(m.group(1)), int(m.group(2))
        fmt = _yearfmt()  # "MD" or "DM"
        if fmt == "MD":
            if a > 12 and 1 <= b <= 12:
                return f"'{m.group(0)}' looks like DD-MM but config expects MM-DD. Try '{b:02d}-{a:02d}'."
        else:  # DM
            if b > 12 and 1 <= a <= 12:
                return f"'{m.group(0)}' looks like MM-DD but config expects DD-MM. Try '{b:02d}-{a:02d}'."
    return None


def _lint_check_invalid_weekday_names(s: str) -> str | None:
    wd_set = set(_WD_ABBR)  # ["mon","tue","wed","thu","fri","sat","sun"]
    for wd in re.findall(r"\b[a-z]{3,}\b", s):
        if wd in wd_set or wd in ("rand", "rand*"):
            continue
        if re.search(rf"(?:^|[\s\+\|,:@-])(w:|@prev-|@next-|last-|1st|2nd|3rd|4th|5th-){wd}\b", s):
            sug = difflib.get_close_matches(wd, list(wd_set), n=1, cutoff=0.6)
            if sug:
                return f"Unknown weekday '{wd}'. Did you mean '{sug[0]}'?"
    return None


def _lint_check_nth_weekday_suffixes(s: str) -> str | None:
    ord_ok = {"1": "1st", "2": "2nd", "3": "3rd", "4": "4th", "5": "5th"}
    for m in re.finditer(r"\b(\d+)(st|nd|rd|th)-([a-z]+)\b", s):
        n, suff, wd = m.group(1), m.group(2), m.group(3)
        if n not in ord_ok:
            return f"Invalid ordinal '{n}{suff}'. Only 1st..5th are supported."
        expect = ord_ok[n]
        if f"{n}{suff}" != expect:
            return f"Did you mean '{expect}-{wd}' instead of '{n}{suff}-{wd}'?"
    return None


def _lint_check_unsat_pure_weekly_and(s: str) -> str | None:
    wd_set = set(_WD_ABBR)
    and_terms = [t.strip() for t in re.split(r"\|", s)]
    for t in and_terms:
        atoms = [a.strip() for a in re.split(r"\+", t)]
        wsets, only_weekly = [], True
        for a in atoms:
            m = re.match(r"^w(?:(/\d+)?):([a-z0-9\-\:\,]+)$", a)
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
                if tok in wd_set:
                    ws.add(tok)
            if not simple:
                only_weekly = False
                break
            if ws:
                wsets.append(ws)
        if only_weekly and wsets and not set.intersection(*wsets):
            return (
                "These anchors joined with '+' don't share any possible date. "
                "If you meant 'either/or', join them with ',' or '|'."
            )
    return None


def _lint_check_backward_quarter_ranges(s: str) -> str | None:
    g = re.search(r"\bq([1-4])\s*\.\.\s*q([1-4])\b", s)
    if g and int(g.group(2)) < int(g.group(1)):
        return (
            "Invalid quarter range 'qX..qY': end quarter precedes start quarter. "
            "Split across the year boundary, e.g., 'q4, q1'."
        )
    return None


def _lint_collect_warnings(s: str) -> list[str]:
    warnings: list[str] = []
    if re.search(r"y:[^|+)]*@t=\d{2}:\d{2},", s):
        warnings.append("Multiple @t times inside a single 'y:' atom; ensure each spec has its own @t or use '|'.")
    return warnings


def lint_anchor_expr(expr: str) -> tuple[str | None, list[str]]:
    # Accept anchors wrapped in single or double quotes and normalize rand-month alias.
    s = _unwrap_quotes(expr or "").strip().lower()
    if len(s) > 1024:
        return ("Anchor expression too long (max 1024 characters).", [])
    # normalize 'mm-rand' → 'rand-mm' for consistency
    s = re.sub(r"\b(\d{2})-rand\b", r"rand-\1", s)
    s = _lint_expand_year_month_aliases(s)
    if not s:
        return None, []

    fatal = _lint_check_weekly_delimiter_contract(s)
    if fatal:
        return fatal, []

    fatal = _lint_check_yearly_segments(s)
    if fatal:
        return fatal, []

    fatal = _lint_check_global_md_dm_confusion(s)
    if fatal:
        return fatal, []

    fatal = _lint_check_invalid_weekday_names(s)
    if fatal:
        return fatal, []

    fatal = _lint_check_nth_weekday_suffixes(s)
    if fatal:
        return fatal, []

    fatal = _lint_check_unsat_pure_weekly_and(s)
    if fatal:
        return fatal, []

    fatal = _lint_check_backward_quarter_ranges(s)
    if fatal:
        return fatal, []

    return None, _lint_collect_warnings(s)



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

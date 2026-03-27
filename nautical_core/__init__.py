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
import importlib
import types
from contextlib import contextmanager
try:
    import fcntl  # POSIX advisory lock
except Exception:
    fcntl = None

_PKG_BASENAME = os.path.basename(os.path.dirname(__file__))
_PKG_DIR = os.path.dirname(__file__)

if not __package__:
    __package__ = (__name__ if __name__ != _PKG_BASENAME else _PKG_BASENAME)

_PKG_IMPORT_ROOT = str(__package__ or _PKG_BASENAME)
_PKG_PROXY = sys.modules.get(_PKG_IMPORT_ROOT)
if _PKG_PROXY is None:
    _PKG_PROXY = types.ModuleType(_PKG_IMPORT_ROOT)
    sys.modules[_PKG_IMPORT_ROOT] = _PKG_PROXY
_PKG_PROXY.__file__ = __file__
_PKG_PROXY.__package__ = _PKG_IMPORT_ROOT
_PKG_PROXY.__path__ = [_PKG_DIR]


def _import_sibling(module_name: str):
    _PKG_PROXY.__dict__.update(globals())
    return importlib.import_module(f"{_PKG_IMPORT_ROOT}.{module_name}")


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

_config_support = _import_sibling("config_support")

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
    return _config_support.env_flag_true(name, env_map=env_map)


def _path_input_error(path_value: str) -> str | None:
    return _config_support.path_input_error(path_value)


def _normalized_abspath(path_value: str) -> str:
    return _config_support.normalized_abspath(path_value)


def _nearest_existing_dir(path_value: str) -> str | None:
    return _config_support.nearest_existing_dir(path_value)


def _world_writable_without_sticky(mode: int) -> bool:
    return _config_support.world_writable_without_sticky(mode)


def _path_safety_error(path_value: str, *, expect_dir: bool = True) -> str | None:
    return _config_support.path_safety_error(path_value, expect_dir=expect_dir)


def _validated_user_dir(
    path_value: str,
    *,
    label: str,
    trust_env: str = "",
    env_map: dict | None = None,
    warn_on_error: bool = True,
) -> str:
    return _config_support.validated_user_dir(
        path_value,
        label=label,
        trust_env=trust_env,
        env_map=env_map,
        warn_on_error=warn_on_error,
    )


def _read_toml(path: str) -> dict:
    return _config_support.read_toml(
        path,
        tomllib_mod=tomllib,
        warn_missing_toml_parser=_warn_missing_toml_parser,
        warn_toml_parse_error=_warn_toml_parse_error,
    )


def _config_paths() -> list[str]:
    return _config_support.config_paths(warn_env_config_missing=_warn_env_config_missing)

def _warn_env_config_missing(env_path: str) -> None:
    _config_support.warn_env_config_missing(
        env_path,
        warn_once_per_day_any=_warn_once_per_day_any,
    )


def _normalize_keys(d: dict) -> dict:
    return _config_support.normalize_keys(d)

def _load_config() -> dict:
    return _config_support.load_config(
        defaults=_DEFAULTS,
        config_paths=_config_paths,
        read_toml=_read_toml,
        normalize_keys=_normalize_keys,
    )



def _nautical_cache_dir() -> str:
    _cache_support = _import_sibling("cache_support")

    return _cache_support.nautical_cache_dir(validated_user_dir=_validated_user_dir)


def _warn_once_per_day(key: str, message: str) -> None:
    _warnings = _import_sibling("warnings")

    _warnings.warn_once_per_day(
        key,
        message,
        cache_dir=_nautical_cache_dir(),
        require_diag=True,
    )


def _warn_once_per_day_any(key: str, message: str) -> None:
    _warnings = _import_sibling("warnings")

    _warnings.warn_once_per_day(
        key,
        message,
        cache_dir=_nautical_cache_dir(),
        require_diag=False,
    )


def _warn_rate_limited_any(key: str, message: str, min_interval_s: float = 3600.0) -> None:
    _warnings = _import_sibling("warnings")

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
_ui = _import_sibling("ui")

strip_rich_markup = _ui.strip_rich_markup
term_width_stderr = _ui.term_width_stderr
fast_color_enabled = _ui.fast_color_enabled
ansi = _ui.ansi
emit_wrapped = _ui.emit_wrapped
emit_line = _ui.emit_line
text_line = _ui.text_line
panel_line_from_rows = _ui.panel_line_from_rows
panel_line = _ui.panel_line
render_panel = _ui.render_panel



def _warn_missing_toml_parser(config_path: str) -> None:
    _warnings = _import_sibling("warnings")

    _warnings.warn_missing_toml_parser(
        config_path,
        warn_once_per_day=_warn_once_per_day,
        warn_once_per_day_any=_warn_once_per_day_any,
    )


def _warn_toml_parse_error(config_path: str, err: Exception) -> None:
    _warnings = _import_sibling("warnings")

    _warnings.warn_toml_parse_error(
        config_path,
        err,
        warn_once_per_day=_warn_once_per_day,
        warn_once_per_day_any=_warn_once_per_day_any,
    )


def _get_config() -> dict:
    global _CONF_CACHE
    out, _CONF_CACHE = _config_support.get_config(_CONF_CACHE, load_config=_load_config)
    return out

_CONF = MappingProxyType(_get_config())

def _conf_raw(key: str):
    return _config_support.conf_raw(_CONF, key)

def _conf_str(key: str, default: str) -> str:
    return _config_support.conf_str(_CONF, key, default)

def _conf_int(
    key: str,
    default: int,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    return _config_support.conf_int(
        _CONF,
        key,
        default,
        min_value=min_value,
        max_value=max_value,
    )

def _conf_bool(
    key: str,
    default: bool = False,
    true_values: set[str] | None = None,
    false_values: set[str] | None = None,
) -> bool:
    return _config_support.conf_bool(
        _CONF,
        key,
        default=default,
        true_values=true_values,
        false_values=false_values,
    )


def _conf_csv_or_list(key: str, default: list[str] | None = None, lower: bool = False) -> list[str]:
    return _config_support.conf_csv_or_list(_CONF, key, default=default, lower=lower)


def _conf_uda_field_list(key: str) -> list[str]:
    return _config_support.conf_uda_field_list(_CONF, key)

def _trueish(v, default=False):
    return _config_support.trueish(v, default=default)

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
_common = _import_sibling("common")
_cache_payload = _import_sibling("cache_payload")
_cache_locking = _import_sibling("cache_locking")
_acf_support = _import_sibling("acf_support")
_cached_expansion = _import_sibling("cached_expansion")
_nth_monthly = _import_sibling("nth_monthly")
_expansion_support = _import_sibling("expansion_support")
_monthly_support = _import_sibling("monthly_support")
_natural_language = _import_sibling("natural_language")
_linting = _import_sibling("linting")
_parser_atoms = _import_sibling("parser_atoms")
_parser_dnf = _import_sibling("parser_dnf")
_parser_frontend = _import_sibling("parser_frontend")
_precompute = _import_sibling("precompute")
_quarter_helpers = _import_sibling("quarter_helpers")
_quarter_rewrite = _import_sibling("quarter_rewrite")
_quarter_selector = _import_sibling("quarter_selector")
_satisfiability = _import_sibling("satisfiability")
_schedule_utils = _import_sibling("schedule_utils")
_scheduler_atom = _import_sibling("scheduler_atom")
_scheduler_expr = _import_sibling("scheduler_expr")
_strict_validation = _import_sibling("strict_validation")
_tokenutil = _import_sibling("tokenutil")
_yearly_parse = _import_sibling("yearly_parse")
_yearly_validation = _import_sibling("yearly_validation")
_year_tokens = _import_sibling("year_tokens")

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

_timeutil = _import_sibling("timeutil")

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
    return _acf_support.atom_sort_key(x, json_mod=json)

# Unpack function your validators call
def _acf_unpack(packed: str) -> dict:
    return _acf_support.acf_unpack(
        packed,
        base64_mod=base64,
        zlib_mod=zlib,
        json_mod=json,
    )

def build_acf(expr: str) -> str:
    return _acf_support.build_acf(
        expr,
        parse_anchor_expr_to_dnf_cached=parse_anchor_expr_to_dnf_cached,
        coerce_int=coerce_int,
        normalize_spec_for_acf=_normalize_spec_for_acf,
        mods_to_acf=_mods_to_acf,
        atom_sort_key=_atom_sort_key,
        json_mod=json,
        zlib_mod=zlib,
        base64_mod=base64,
        hashlib_mod=hashlib,
        acf_checksum_len=ACF_CHECKSUM_LEN,
    )

def _normalize_spec_for_acf_uncached(typ: str, spec: str):
    return _acf_support.normalize_spec_for_acf_uncached(
        typ,
        spec,
        expand_weekly_aliases=_expand_weekly_aliases,
        split_csv_tokens=_split_csv_tokens,
        normalize_weekday=_normalize_weekday,
        expand_monthly_aliases=_expand_monthly_aliases,
        re_mod=re,
        year_pair=_year_pair,
    )


@_ttl_lru_cache(maxsize=512)
def _normalize_spec_for_acf_cached(typ: str, spec: str, fmt: str):
    typ = (typ or "").strip().lower()[:1]
    if typ not in ("w", "m", "y"):
        return None
    spec = (spec or "").strip().lower()[:256]
    fmt = "DM" if (fmt or "").upper() == "DM" else "MD"
    return _normalize_spec_for_acf_uncached(typ, spec)


def _normalize_spec_for_acf(typ: str, spec: str):
    return _acf_support.normalize_spec_for_acf(
        typ,
        spec,
        normalize_spec_for_acf_cached=lambda t, s: _normalize_spec_for_acf_cached(t, s, _yearfmt()),
        clone_mod_value=_clone_mod_value,
    )

def is_valid_acf(acf_str: str) -> bool:
    return _acf_support.is_valid_acf(
        acf_str,
        hashlib_mod=hashlib,
        acf_checksum_len=ACF_CHECKSUM_LEN,
        acf_unpack=_acf_unpack,
    )



def acf_to_original_format(acf_str: str) -> str:
    return _acf_support.acf_to_original_format(
        acf_str,
        is_valid_acf=is_valid_acf,
        acf_unpack=_acf_unpack,
        acf_spec_to_string=_acf_spec_to_string,
        acf_mods_to_string=_acf_mods_to_string,
    )


@_ttl_lru_cache(maxsize=512)
def _year_pair_cached(a: int, b: int, fmt: str) -> tuple[int, int]:
    """Interpret (a,b) according to ANCHOR_YEAR_FMT; return (day, month)."""
    return (b, a) if fmt == "MD" else (a, b)


def _year_pair(a: int, b: int) -> tuple[int, int]:
    return _year_pair_cached(a, b, _yearfmt())

def _mods_to_acf(mods: dict) -> dict:
    return _acf_support.mods_to_acf(mods, hhmm_re=_hhmm_re)

def _acf_mods_to_string(m: dict) -> str:
    return _acf_support.acf_mods_to_string(m, wd_abbr=_WD_ABBR)

def _acf_spec_to_string(typ: str, spec) -> str:
    return _acf_support.acf_spec_to_string(
        typ,
        spec,
        tok=_tok,
        tok_range=_tok_range,
    )


# ==============================================================================
# SECTION: Anchor cache & locking
# ==============================================================================
# --- Cache directory discovery & IO ---
_CACHE_DIR = None

def _cache_dir() -> str:
    global _CACHE_DIR
    _cache_support = _import_sibling("cache_support")

    chosen = _cache_locking.cache_dir(
        _CACHE_DIR,
        anchor_cache_dir_override=ANCHOR_CACHE_DIR_OVERRIDE,
        nautical_cache_dir_path=_nautical_cache_dir(),
        validated_user_dir=_validated_user_dir,
        select_cache_dir=_cache_support.select_cache_dir,
    )
    _CACHE_DIR = chosen
    return chosen

def _cache_key(acf: str, anchor_mode: str) -> str:
    _cache_support = _import_sibling("cache_support")

    return _cache_support.cache_key(
        acf,
        anchor_mode,
        anchor_year_fmt=ANCHOR_YEAR_FMT,
        wrand_salt=WRAND_SALT,
        local_tz_name=LOCAL_TZ_NAME,
        holiday_region=HOLIDAY_REGION,
    )

def _cache_path(key: str) -> str:
    _cache_support = _import_sibling("cache_support")

    return _cache_support.cache_path(_cache_dir(), key)

def _cache_lock_path(key: str) -> str:
    _cache_support = _import_sibling("cache_support")

    return _cache_support.cache_lock_path(_cache_dir(), key)

@contextmanager
def _safe_lock_sleep_once(sleep_base: float, jitter: float) -> None:
    _cache_locking.safe_lock_sleep_once(
        sleep_base,
        jitter,
        time_mod=time,
        random_mod=random,
    )


def _safe_lock_ensure_parent(path_str: str, mkdir: bool) -> None:
    _cache_locking.safe_lock_ensure_parent(path_str, mkdir, os_mod=os)


def _safe_lock_age(path_str: str) -> float | None:
    return _cache_locking.safe_lock_age(path_str, time_mod=time, os_mod=os)


def _safe_lock_stale_pid(path_str: str, stale_after: float | None) -> bool:
    return _cache_locking.safe_lock_stale_pid(
        path_str,
        stale_after,
        time_mod=time,
        os_mod=os,
    )


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
    with _cache_locking.safe_lock_fcntl_context(
        path_str,
        tries=tries,
        sleep_base=sleep_base,
        jitter=jitter,
        mode=mode,
        mkdir=mkdir,
        safe_lock_ensure_parent=_safe_lock_ensure_parent,
        safe_lock_sleep_once=_safe_lock_sleep_once,
        fcntl_mod=fcntl,
        os_mod=os,
    ) as acquired:
        yield acquired


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
    with _cache_locking.safe_lock_excl_context(
        path_str,
        tries=tries,
        sleep_base=sleep_base,
        jitter=jitter,
        mode=mode,
        mkdir=mkdir,
        stale_after=stale_after,
        safe_lock_ensure_parent=_safe_lock_ensure_parent,
        safe_lock_stale_pid=_safe_lock_stale_pid,
        safe_lock_age=_safe_lock_age,
        safe_lock_sleep_once=_safe_lock_sleep_once,
        os_mod=os,
        time_mod=time,
    ) as acquired:
        yield acquired


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
    with _cache_locking.safe_lock(
        path,
        retries=retries,
        sleep_base=sleep_base,
        jitter=jitter,
        mode=mode,
        mkdir=mkdir,
        stale_after=stale_after,
        fcntl_mod=fcntl,
        os_mod=os,
        time_mod=time,
        random_mod=random,
    ) as acquired:
        yield acquired

@contextmanager
def _cache_lock(key: str):
    with _cache_locking.cache_lock(
        key,
        cache_lock_path=_cache_lock_path,
        safe_lock=safe_lock,
        cache_lock_retries=_CACHE_LOCK_RETRIES,
        cache_lock_sleep_base=_CACHE_LOCK_SLEEP_BASE,
        cache_lock_jitter=_CACHE_LOCK_JITTER,
        cache_lock_stale_after=_CACHE_LOCK_STALE_AFTER,
    ) as acquired:
        yield acquired


# ==============================================================================
# SECTION: Hook utilities (diag, run_task)
# ==============================================================================
_runtime = _import_sibling("runtime")

_DIAG_LOG_REDACT_KEYS: frozenset[str] = _runtime.DIAG_LOG_REDACT_KEYS
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
    return _cache_payload.is_dnf_like(dnf, is_atom_like=_is_atom_like)

def _is_atom_like(atom) -> bool:
    return _cache_payload.is_atom_like(atom)


def _clone_mod_value(v):
    return _cache_payload.clone_mod_value(v)


def _clone_mods(mods):
    return _cache_payload.clone_mods(mods)


def _clone_atom(atom):
    return _cache_payload.clone_atom(atom)


def _clone_dnf(dnf):
    return _cache_payload.clone_dnf(dnf)


def _clone_cache_payload(obj: dict) -> dict:
    return _cache_payload.clone_cache_payload(obj)


def _normalize_dnf_cached(dnf):
    return _cache_payload.normalize_dnf_cached(dnf)


def _cache_payload_shape_ok(obj: dict) -> bool:
    return _cache_payload.cache_payload_shape_ok(obj, is_dnf_like=_is_dnf_like)


def _cache_atomic_replace(src: str, dst: str) -> None:
    _cache_payload.cache_atomic_replace(src, dst, os_mod=os)

def cache_load(key: str) -> dict | None:
    return _cache_payload.cache_load(
        key,
        enable_anchor_cache=ENABLE_ANCHOR_CACHE,
        cache_path=_cache_path,
        anchor_cache_ttl=ANCHOR_CACHE_TTL,
        time_mod=time,
        cache_load_mem=_CACHE_LOAD_MEM,
        cache_load_mem_ttl=_CACHE_LOAD_MEM_TTL,
        clone_cache_payload=_clone_cache_payload,
        normalize_dnf_cached=_normalize_dnf_cached,
        cache_payload_shape_ok=_cache_payload_shape_ok,
        cache_load_mem_max=_CACHE_LOAD_MEM_MAX,
        diag=diag,
        os_mod=os,
        json_mod=json,
        zlib_mod=zlib,
        base64_mod=base64,
    )

def cache_save(key: str, obj: dict) -> bool:
    return _cache_payload.cache_save(
        key,
        obj,
        enable_anchor_cache=ENABLE_ANCHOR_CACHE,
        json_mod=json,
        zlib_mod=zlib,
        base64_mod=base64,
        cache_path=_cache_path,
        cache_dir=_cache_dir,
        cache_lock=_cache_lock,
        diag=diag,
        os_mod=os,
        tempfile_mod=tempfile,
        cache_atomic_replace=_cache_atomic_replace,
        cache_load_mem=_CACHE_LOAD_MEM,
    )

@_ttl_lru_cache(maxsize=1024)
def _cache_key_for_task_cached(anchor_expr: str, anchor_mode: str, fmt: str) -> str:
    return _cache_payload.cache_key_for_task_cached(
        anchor_expr,
        anchor_mode,
        fmt,
        build_acf=build_acf,
        cache_key=_cache_key,
    )


def cache_key_for_task(anchor_expr: str, anchor_mode: str) -> str:
    return _cache_key_for_task_cached(anchor_expr or "", anchor_mode or "", _yearfmt())


# ---- Core iterator over DNF ---------------------------------------------------
_NTH_RE  = re.compile(r"^(?:(\d)(?:st|nd|rd|th)|last)-(" + "|".join(_WD_ABBR) + r")$")

def _days_in_month(y:int, m:int) -> int:
    return _expansion_support.days_in_month(y, m, monthrange=monthrange)

def _wd_idx(s: str) -> int | None:
    return _expansion_support.wd_idx(s, wd_abbr=_WD_ABBR)


@_ttl_lru_cache(maxsize=128)
def _wday_idx_any(s: str) -> int | None:
    return _expansion_support.wday_idx_any(s, weekdays=_WEEKDAYS, wd_idx=_wd_idx)


def _weekly_spec_to_wset(spec: str, mods: dict | None = None) -> set[int]:
    return _expansion_support.weekly_spec_to_wset(
        spec,
        mods=mods,
        expand_weekly_aliases=_expand_weekly_aliases,
        split_csv_lower=_split_csv_lower,
        wday_idx_any=_wday_idx_any,
    )

def _doms_for_weekly_spec(spec:str, y:int, m:int) -> set[int]:
    return _expansion_support.doms_for_weekly_spec(
        spec,
        y,
        m,
        expand_weekly_aliases=_expand_weekly_aliases,
        split_csv_tokens=_split_csv_tokens,
        wd_idx=_wd_idx,
        days_in_month=_days_in_month,
    )

def _doms_for_monthly_token(tok: str, y:int, m:int) -> set[int]:
    return _monthly_support.doms_for_monthly_token(
        tok,
        y,
        m,
        monthly_alias=_MONTHLY_ALIAS,
        days_in_month=_days_in_month,
        re_mod=re,
        nth_re=_NTH_RE,
        wd_idx=_wd_idx,
    )

def _y_ranges_from_spec(spec: str) -> list[tuple[int,int,int,int]]:
    return _expansion_support.y_ranges_from_spec(
        spec,
        split_csv_lower=_split_csv_lower,
        re_mod=re,
        year_pair=_year_pair,
    )


def _doms_allowed_by_year(y:int, m:int, y_specs: list[str]) -> set[int]:
    return _expansion_support.doms_allowed_by_year(
        y,
        m,
        y_specs,
        y_ranges_from_spec=_y_ranges_from_spec,
        days_in_month=_days_in_month,
    )

def _month_allowed_doms_for_monthly_atom(atom: dict, y: int, m: int, dim: int) -> set[int]:
    return _monthly_support.month_allowed_doms_for_monthly_atom(
        atom,
        y,
        m,
        dim,
        split_csv_lower=_split_csv_lower,
        doms_for_monthly_token=_doms_for_monthly_token,
    )


def _intersect_monthly_atoms_allowed(
    term: list[dict],
    *,
    y: int,
    m: int,
    dim: int,
    allowed: set[int],
) -> set[int]:
    return _monthly_support.intersect_monthly_atoms_allowed(
        term,
        y=y,
        m=m,
        dim=dim,
        allowed=allowed,
        month_allowed_doms_for_monthly_atom=_month_allowed_doms_for_monthly_atom,
    )


def _next_for_and_rand_yearly(term: list[dict], ref_d: date, y_specs: list[str]) -> date | None:
    return _scheduler_expr.next_for_and_rand_yearly(
        term,
        ref_d,
        y_specs,
        wrand_salt=WRAND_SALT,
        days_in_month=_days_in_month,
        doms_allowed_by_year=_doms_allowed_by_year,
        intersect_monthly_atoms_allowed=_intersect_monthly_atoms_allowed,
        doms_for_weekly_spec=_doms_for_weekly_spec,
        date_cls=date,
    )


def _next_for_and_fast_path(term: list[dict], ref_d: date, seed: date) -> date:
    return _scheduler_expr.next_for_and_fast_path(
        term,
        ref_d,
        seed,
        next_after_atom_with_mods=next_after_atom_with_mods,
        atom_matches_on=atom_matches_on,
        max_anchor_iter=MAX_ANCHOR_ITER,
        warn_once_per_day=_warn_once_per_day,
        parse_error_cls=ParseError,
        os_mod=os,
    )


def _next_for_and(term: list[dict], ref_d: date, seed: date) -> date:
    return _scheduler_expr.next_for_and(
        term,
        ref_d,
        seed,
        wrand_salt=WRAND_SALT,
        days_in_month=_days_in_month,
        doms_allowed_by_year=_doms_allowed_by_year,
        intersect_monthly_atoms_allowed=_intersect_monthly_atoms_allowed,
        doms_for_weekly_spec=_doms_for_weekly_spec,
        next_after_atom_with_mods=next_after_atom_with_mods,
        atom_matches_on=atom_matches_on,
        max_anchor_iter=MAX_ANCHOR_ITER,
        warn_once_per_day=_warn_once_per_day,
        parse_error_cls=ParseError,
        os_mod=os,
        date_cls=date,
    )


def _next_for_or(dnf: list[list[dict]], ref_d: date, seed: date) -> date:
    return _scheduler_expr.next_for_or(dnf, ref_d, seed, next_for_and=_next_for_and)

# ---- Public precompute --------------------------------------------------------

def precompute_hints(dnf: list[list[dict]],
                     start_dt: datetime | None = None,
                     anchor_mode: str = "ALL",
                     rand_seed: str | None = None,
                     k_next: int = 24,
                     sample_days_for_year: int = 366) -> dict:
    _ = anchor_mode
    return _precompute.precompute_hints(
        dnf,
        start_dt=start_dt,
        rand_seed=rand_seed,
        k_next=k_next,
        sample_days_for_year=sample_days_for_year,
        now_local=datetime.now,
        next_after_expr=next_after_expr,
        next_for_or=_next_for_or,
    )


# ───────────────── Cache writer ───────────────── 

def build_and_cache_hints(anchor_expr: str,
                          anchor_mode: str = "ALL",
                          default_due_dt=None) -> AnchorHintsPayload:
    return cast(
        AnchorHintsPayload,
        _precompute.build_and_cache_hints(
            anchor_expr,
            anchor_mode=anchor_mode,
            default_due_dt=default_due_dt,
            cache_key_for_task=cache_key_for_task,
            cache_load=cache_load,
            validate_anchor_expr_strict=validate_anchor_expr_strict,
            describe_anchor_expr_from_dnf=_describe_anchor_expr_from_dnf,
            precompute_hints=precompute_hints,
            cache_save=cache_save,
            anchor_year_fmt=ANCHOR_YEAR_FMT,
            wrand_salt=WRAND_SALT,
            local_tz_name=LOCAL_TZ_NAME,
            holiday_region=HOLIDAY_REGION,
        ),
    )


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


def _ordinal(n: int) -> str:
    return _natural_language.ordinal(n)


def _term_collect_mods(term: list) -> dict:
    return _natural_language.term_collect_mods(term)


def _fmt_hhmm_for_term(term: list, default_due_dt):
    return _natural_language.fmt_hhmm_for_term(term, default_due_dt)


def _fmt_weekdays_list(spec: str) -> str:
    return _natural_language.fmt_weekdays_list(
        spec,
        expand_weekly_aliases=_expand_weekly_aliases,
        split_csv_lower=_split_csv_lower,
        wday_idx_any=_wday_idx_any,
    )


def _fmt_monthly_atom(spec: str) -> str:
    return _natural_language.fmt_monthly_atom(
        spec,
        monthly_alias=_MONTHLY_ALIAS,
        safe_match=_safe_match,
        nth_wd_re=_nth_wd_re,
        bd_re=_bd_re,
    )


def _fmt_md(d: int, m: int) -> str:
    fmt = (globals().get("ANCHOR_YEAR_FMT") or "DM").upper()
    name = _natural_language._MONTH_ABBR[m - 1]
    return f"{d} {name}" if fmt == "DM" else f"{name} {d}"


def _is_full_month(d1, m1, d2, m2) -> int | None:
    """Return month number if token covers the whole month, else None.
    Accepts 28..31 as 'end of month' (Feb handled leniently)."""
    if m1 != m2 or d1 != 1:
        return None
    return m1 if 28 <= d2 <= 31 else None


def _fmt_yearly_atom(tok: str) -> str:
    return _natural_language.fmt_yearly_atom(
        tok,
        rand_mm_re=_rand_mm_re,
        md_range_re=_md_range_re,
        yearfmt=_yearfmt,
    )




def _describe_monthly_tokens(spec: str):
    return _natural_language.describe_monthly_tokens(spec, split_csv_lower=_split_csv_lower)


def _describe_is_pure_nth_weekday_spec(spec: str):
    return _natural_language.describe_is_pure_nth_weekday_spec(
        spec,
        split_csv_lower=_split_csv_lower,
        safe_match=_safe_match,
        nth_wd_re=_nth_wd_re,
    )


def _describe_is_pure_dom_spec(spec: str):
    return _natural_language.describe_is_pure_dom_spec(spec, split_csv_lower=_split_csv_lower)


def _describe_single_full_month_from_yearly_spec(spec: str):
    return _natural_language.describe_single_full_month_from_yearly_spec(
        spec,
        year_range_colon_re=_year_range_colon_re,
    )


def _describe_term_roll_shift(term) -> str | None:
    return _natural_language.describe_term_roll_shift(term)


def _describe_term_bd_filter(term) -> bool:
    return _natural_language.describe_term_bd_filter(term)


def _describe_roll_suffix(roll: str) -> str:
    return _natural_language.describe_roll_suffix(roll)


def _describe_inject_schedule_suffixes(txt: str, term) -> str:
    return _natural_language.describe_inject_schedule_suffixes(txt, term)


def _describe_anchor_term_collect(term):
    return _natural_language.describe_anchor_term_collect(
        term,
        fmt_weekdays_list=_fmt_weekdays_list,
        split_csv_tokens=_split_csv_tokens,
        fmt_monthly_atom=_fmt_monthly_atom,
        fmt_yearly_atom=_fmt_yearly_atom,
    )


def _describe_anchor_term_fused_month_year(
    term,
    default_due_dt,
    monthly_specs,
    yearly_specs,
    yr_ival: int,
    bd_filter: bool,
    m_parts: list[str],
) -> str | None:
    return _natural_language.describe_anchor_term_fused_month_year(
        term,
        default_due_dt,
        monthly_specs,
        yearly_specs,
        yr_ival,
        bd_filter,
        m_parts,
        describe_is_pure_nth_weekday_spec=_describe_is_pure_nth_weekday_spec,
        describe_single_full_month_from_yearly_spec=_describe_single_full_month_from_yearly_spec,
        fmt_hhmm_for_term=_fmt_hhmm_for_term,
    )


def _describe_anchor_term_interval_prefix(wk_ival, mo_ival, yr_ival, monthly_specs):
    return _natural_language.describe_anchor_term_interval_prefix(
        wk_ival,
        mo_ival,
        yr_ival,
        monthly_specs,
        describe_is_pure_nth_weekday_spec=_describe_is_pure_nth_weekday_spec,
        describe_is_pure_dom_spec=_describe_is_pure_dom_spec,
    )


def _describe_anchor_term_parts(w_phrase, m_parts, y_parts, bd_filter: bool) -> list[str]:
    return _natural_language.describe_anchor_term_parts(w_phrase, m_parts, y_parts, bd_filter)


def describe_anchor_term(term: list, default_due_dt=None) -> str:
    return _natural_language.describe_anchor_term(
        term,
        default_due_dt=default_due_dt,
        fmt_weekdays_list=_fmt_weekdays_list,
        split_csv_tokens=_split_csv_tokens,
        fmt_monthly_atom=_fmt_monthly_atom,
        fmt_yearly_atom=_fmt_yearly_atom,
        describe_is_pure_nth_weekday_spec=_describe_is_pure_nth_weekday_spec,
        describe_single_full_month_from_yearly_spec=_describe_single_full_month_from_yearly_spec,
        fmt_hhmm_for_term=_fmt_hhmm_for_term,
        describe_is_pure_dom_spec=_describe_is_pure_dom_spec,
    )

def _describe_anchor_expr_from_dnf(dnf: list, default_due_dt=None) -> str:
    return _natural_language.describe_anchor_expr_from_dnf(
        dnf,
        default_due_dt=default_due_dt,
        describe_anchor_term=describe_anchor_term,
    )


def describe_anchor_expr(anchor_expr: str, default_due_dt=None) -> str:
    return _natural_language.describe_anchor_expr(
        anchor_expr,
        default_due_dt=default_due_dt,
        parse_anchor_expr_to_dnf_cached=parse_anchor_expr_to_dnf_cached,
        describe_anchor_expr_from_dnf=_describe_anchor_expr_from_dnf,
    )



def _term_prevnext_wd(term):
    return _natural_language.term_prevnext_wd(term, wdname=_natural_language._WDNAME)


def _inject_prevnext_phrase(txt: str, term) -> str:
    return _natural_language.inject_prevnext_phrase(
        txt,
        term,
        wdname=_natural_language._WDNAME,
    )


def _join_natural_or_terms(terms: list[str]) -> str:
    return _natural_language.join_natural_or_terms(terms)


def _longest_common_suffix(parts: list[str]) -> str:
    return _natural_language.longest_common_suffix(parts)


def _compress_or_terms_by_clause(terms: list[str], delim: str) -> str | None:
    return _natural_language.compress_or_terms_by_clause(terms, delim)


def describe_anchor_dnf(dnf: list, task: dict) -> str:
    return _natural_language.describe_anchor_dnf(
        dnf,
        task,
        try_bucket_rand_monthly=_try_bucket_rand_monthly,
        parse_dt_any=parse_dt_any,
        describe_anchor_term=describe_anchor_term,
    )


def _normalize_range_token(tok: str) -> str | None:
    return _natural_language.normalize_range_token(
        tok,
        safe_match=_safe_match,
        int_range_re=_int_range_re,
    )


def _rand_bucket_time_from_mods(mods: dict) -> str | None:
    return _natural_language.rand_bucket_time_from_mods(mods)


def _rand_bucket_merge_mods(mods: dict, time_str: str | None, bd_flag: bool) -> tuple[str | None, bool]:
    return _natural_language.rand_bucket_merge_mods(mods, time_str, bd_flag)


def _rand_bucket_signature(term: list[dict]) -> tuple | None:
    return _natural_language.rand_bucket_signature(
        term,
        normalize_range_token=_normalize_range_token,
    )


def _try_bucket_rand_monthly(dnf: list[list[dict]], task: dict) -> str | None:
    return _natural_language.try_bucket_rand_monthly(
        dnf,
        task,
        rand_bucket_signature=_rand_bucket_signature,
    )


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

_dates = _import_sibling("dates")


def month_len(y, m):
    return _dates.month_len(y, m)


def add_months(d: date, months: int) -> date:
    return _dates.add_months(d, months)


def months_days_between(d1: date, d2: date):
    return _dates.months_days_between(d1, d2)


def humanize_delta(from_dt: datetime, to_dt: datetime, use_months_days: bool):
    return _dates.humanize_delta(from_dt, to_dt, use_months_days)


def _active_mod_keys(mods: dict) -> set:
    return _cached_expansion.active_mod_keys(mods)


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
    return _cached_expansion.week_monday(d)


def _seeded_int(key: str) -> int:
    return _cached_expansion.seeded_int(key)


def _weekly_rand_pick(iso_year: int, iso_week: int, mods: dict) -> int:
    return _cached_expansion.weekly_rand_pick(
        iso_year,
        iso_week,
        mods,
        wrand_salt=WRAND_SALT,
        seeded_int=_seeded_int,
    )


def _is_bd(dt: _date):  # business day
    return _cached_expansion.is_bd(dt)


def _sha_pick(seq_len: int, seed_key: str) -> int:
    return _cached_expansion.sha_pick(seq_len, seed_key)


def _term_rand_info(term):
    return _cached_expansion.term_rand_info(term)



def _filter_by_w(dt_list: list[_date], term: list[dict]):
    return _cached_expansion.filter_by_w(
        dt_list,
        term,
        atype=_atype,
        aspec=_aspec,
        weekly_spec_to_wset=_weekly_spec_to_wset,
    )


@_ttl_lru_cache(maxsize=128)
def _month_tokens_for_atom_cached(y: int, m: int, spec: str) -> set[int]:
    return _cached_expansion.month_tokens_for_atom_values(
        y,
        m,
        spec,
        expand_monthly_aliases=_expand_monthly_aliases,
        days_in_month=_days_in_month,
        bd_re=_bd_re,
        nth_weekday_re=_nth_weekday_re,
        weekday_map=_WD,
        re_mod=re,
    )


def _month_tokens_for_atom(a: dict, y: int, m: int) -> set[int]:
    return _cached_expansion.month_tokens_for_atom(
        a,
        y,
        m,
        month_tokens_for_atom_cached=_month_tokens_for_atom_cached,
    )


def _term_candidates_in_month(
    term: list[dict], y: int, m: int, rand_atom_idx: int, bd_only: bool
):
    return _cached_expansion.term_candidates_in_month(
        term,
        y,
        m,
        rand_atom_idx,
        bd_only,
        days_in_month=_days_in_month,
        is_bd=_is_bd,
        filter_by_w=_filter_by_w,
        atype=_atype,
        aspec=_aspec,
        month_tokens_for_atom=_month_tokens_for_atom,
        doms_allowed_by_year=_doms_allowed_by_year,
    )


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
    return _parser_atoms.parse_hhmm(s, hhmm_re=_hhmm_re)


def _parse_atom_head(head: str) -> tuple[str, int]:
    return _parser_atoms.parse_atom_head(head, re_mod=re, parse_error_cls=ParseError)




def _parse_atom_mods(mods_str: str):
    return _parser_atoms.parse_atom_mods(
        mods_str,
        split_csv_tokens=_split_csv_tokens,
        parse_hhmm=_parse_hhmm,
        next_prev_wd_re=_next_prev_wd_re,
        weekdays=_WEEKDAYS,
        day_offset_re=_day_offset_re,
        parse_error_cls=ParseError,
    )


@_ttl_lru_cache(maxsize=512)
def _parse_y_token_cached(tok: str, fmt: str):
    return _yearly_parse.parse_y_token(
        tok,
        fmt,
        quarters=_QUARTERS,
        months=_MONTHS,
        y_token_re=_y_token_re,
        re_mod=re,
    )


def _parse_y_token(tok: str):
    return _parse_y_token_cached(tok, _yearfmt())


# ------------------------------------------------------------------------------
# Anchor DNF parser
# ------------------------------------------------------------------------------
def _normalize_anchor_expr_input(s: str) -> str:
    return _parser_frontend.normalize_anchor_expr_input(
        s,
        unwrap_quotes=_unwrap_quotes,
        rewrite_weekly_multi_time_atoms=_rewrite_weekly_multi_time_atoms,
        re_mod=re,
        parse_error_cls=ParseError,
    )


def _fatal_bad_colon_in_year_tail(tail: str) -> str | None:
    return _parser_frontend.fatal_bad_colon_in_year_tail(
        tail,
        split_csv_tokens=_split_csv_tokens,
        re_mod=re,
        yearfmt=_yearfmt,
    )


def _raise_on_bad_colon_year_tokens(s: str) -> None:
    _parser_frontend.raise_on_bad_colon_year_tokens(
        s,
        re_mod=re,
        fatal_bad_colon_in_year_tail=_fatal_bad_colon_in_year_tail,
        parse_error_cls=ParseError,
    )


def _skip_ws_pos(s: str, i: int, n: int) -> int:
    return _parser_frontend.skip_ws_pos(s, i, n)


def _raise_if_comma_joined_anchors(full_tail: str) -> None:
    _parser_frontend.raise_if_comma_joined_anchors(
        full_tail,
        re_mod=re,
        parse_error_cls=ParseError,
    )


def _normalize_monthly_ordinal_spec(spec: str) -> str:
    return _parser_atoms.normalize_monthly_ordinal_spec(spec, re_mod=re)


def _build_anchor_atom_dnf(head: str, full_tail: str):
    return _parser_atoms.build_anchor_atom_dnf(
        head,
        full_tail,
        parse_atom_head=_parse_atom_head,
        parse_group_with_inline_mods=_parse_group_with_inline_mods,
        normalize_monthly_ordinal_spec=_normalize_monthly_ordinal_spec,
        split_csv_lower=_split_csv_lower,
        parse_atom_mods=_parse_atom_mods,
    )


def _parse_anchor_atom_at(s: str, i: int, n: int):
    return _parser_atoms.parse_anchor_atom_at(
        s,
        i,
        n,
        skip_ws_pos=_skip_ws_pos,
        raise_if_comma_joined_anchors=_raise_if_comma_joined_anchors,
        build_anchor_atom_dnf=_build_anchor_atom_dnf,
        parse_error_cls=ParseError,
    )


def parse_anchor_expr_to_dnf(s: str) -> AnchorDNF:
    return _parser_dnf.parse_anchor_expr_to_dnf(
        s,
        normalize_anchor_expr_input=_normalize_anchor_expr_input,
        raise_on_bad_colon_year_tokens=_raise_on_bad_colon_year_tokens,
        parse_anchor_atom_at=_parse_anchor_atom_at,
        skip_ws_pos=_skip_ws_pos,
        rewrite_quarters_in_context=_rewrite_quarters_in_context,
        rewrite_year_month_aliases_in_context=_rewrite_year_month_aliases_in_context,
        validate_year_tokens_in_dnf=_validate_year_tokens_in_dnf,
        validate_and_terms_satisfiable=_validate_and_terms_satisfiable,
        max_anchor_dnf_terms=MAX_ANCHOR_DNF_TERMS,
        parse_error_cls=ParseError,
        today=date.today,
    )


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
    return _yearly_validation.yearly_pair_from_fmt(a, b, fmt)


def _yearly_mmdd_error(mm: int, dd: int) -> str | None:
    return _yearly_validation.yearly_mmdd_error(mm, dd)


def _validate_yearly_token_allowlist(tok: str, fmt: str) -> None:
    _yearly_validation.validate_yearly_token_allowlist(
        tok,
        fmt,
        year_token_format_error_cls=YearTokenFormatError,
    )


def _validate_yearly_token_detailed(tok: str, fmt: str) -> tuple[str, str] | None:
    return _yearly_validation.validate_yearly_token_detailed(
        tok,
        fmt,
        year_token_format_error_cls=YearTokenFormatError,
    )


def _validate_yearly_token_format(spec: str):
    _yearly_validation.validate_yearly_token_format(
        spec,
        yearfmt=_yearfmt,
        split_csv_lower=_split_csv_lower,
        year_token_format_error_cls=YearTokenFormatError,
    )


def _validate_year_tokens_in_dnf(dnf):
    _yearly_validation.validate_year_tokens_in_dnf(
        dnf,
        validate_yearly_token_format=_validate_yearly_token_format,
    )


# ---- AND-term satisfiability guard -----------------------------------------
class AndTermUnsatisfiable(ParseError):
    pass


_LEAP_YEAR_FOR_CHECKS = 2028


def _weekday_set_from_weekly_atom(a) -> set[int]:
    return _satisfiability.weekday_set_from_weekly_atom(
        a,
        weekly_spec_to_wset=_weekly_spec_to_wset,
    )


def _md_pairs_from_yearly_spec(spec: str) -> set[tuple[int, int]]:
    return _satisfiability.md_pairs_from_yearly_spec(
        spec,
        expand_yearly_cached=expand_yearly_cached,
        leap_year_for_checks=_LEAP_YEAR_FOR_CHECKS,
    )


def _quick_weekly_and_check(term: list[dict]) -> None:
    _satisfiability.quick_weekly_and_check(
        term,
        weekday_set_from_weekly_atom=_weekday_set_from_weekly_atom,
        and_term_unsatisfiable_cls=AndTermUnsatisfiable,
    )


def _quick_yearly_and_check(term: list[dict]) -> None:
    _satisfiability.quick_yearly_and_check(
        term,
        md_pairs_from_yearly_spec=_md_pairs_from_yearly_spec,
        and_term_unsatisfiable_cls=AndTermUnsatisfiable,
    )


def _term_has_any_match_within(
    term: list[dict], start: date, seed: date, years: int = 8
) -> bool:
    return _satisfiability.term_has_any_match_within(
        term,
        start,
        seed,
        atom_matches_on=atom_matches_on,
        years=years,
    )


# ------------------------------------------------------------------------------
# Anchor satisfiability checks
# ------------------------------------------------------------------------------
def _validate_and_terms_satisfiable(dnf: list[list[dict]], ref_d: date):
    _satisfiability.validate_and_terms_satisfiable(
        dnf,
        ref_d,
        quick_weekly_and_check=_quick_weekly_and_check,
        quick_yearly_and_check=_quick_yearly_and_check,
        term_has_any_match_within=_term_has_any_match_within,
        normalize_spec_for_acf=_normalize_spec_for_acf,
        and_term_unsatisfiable_cls=AndTermUnsatisfiable,
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
    _yearly_validation.validate_yearly_token(
        tok,
        quarters=_QUARTERS,
        parse_y_token=_parse_y_token,
        parse_error_cls=ParseError,
    )


def _yearly_last_day(mm: int) -> int:
    return _yearly_validation.yearly_last_day(mm)


def _yearly_check_day_month(dd: int, mm: int, label: str, tok: str) -> None:
    _yearly_validation.yearly_check_day_month(
        dd,
        mm,
        label,
        tok,
        parse_error_cls=ParseError,
        month_full=_natural_language._MONTH_FULL,
    )


def _validate_yearly_spec_token(tok: str) -> None:
    _yearly_validation.validate_yearly_spec_token(
        tok,
        parse_error_cls=ParseError,
        month_full=_natural_language._MONTH_FULL,
    )


def _validate_yearly_spec(spec: str):
    _yearly_validation.validate_yearly_spec(
        spec,
        split_csv_lower=_split_csv_lower,
        validate_yearly_spec_token=_validate_yearly_spec_token,
        parse_error_cls=ParseError,
    )


def _normalize_anchor_input_to_dnf(expr) -> AnchorDNF:
    return _strict_validation.normalize_anchor_input_to_dnf(
        expr,
        parse_anchor_expr_to_dnf_cached=parse_anchor_expr_to_dnf_cached,
        parse_error_cls=ParseError,
    )


def _assert_dnf_structure_strict(dnf):
    _strict_validation.assert_dnf_structure_strict(
        dnf,
        is_atom_like=_is_atom_like,
        parse_error_cls=ParseError,
    )


def _validate_anchor_atom_strict(a: dict) -> None:
    _strict_validation.validate_anchor_atom_strict(
        a,
        validate_weekly_spec=_validate_weekly_spec,
        validate_monthly_spec=_validate_monthly_spec,
        active_mod_keys=_active_mod_keys,
        validate_yearly_token_format=_validate_yearly_token_format,
        parse_error_cls=ParseError,
    )


def _validate_anchor_dnf_atoms_strict(dnf: AnchorDNF) -> None:
    _strict_validation.validate_anchor_dnf_atoms_strict(
        dnf,
        validate_anchor_atom_strict=_validate_anchor_atom_strict,
    )


def validate_anchor_expr_strict(expr) -> AnchorDNF:
    return _strict_validation.validate_anchor_expr_strict(
        expr,
        normalize_anchor_input_to_dnf=_normalize_anchor_input_to_dnf,
        assert_dnf_structure_strict=_assert_dnf_structure_strict,
        validate_anchor_dnf_atoms_strict=_validate_anchor_dnf_atoms_strict,
    )


# -------- Cached Expansion Functions ----------
@_ttl_lru_cache(maxsize=128)
def expand_weekly_cached(spec: str):
    return _cached_expansion.expand_weekly(
        spec,
        weekly_spec_to_wset=_weekly_spec_to_wset,
    )


@_ttl_lru_cache(maxsize=128)
def expand_weekly_cached_mods(spec: str, bd_only: bool):
    return _cached_expansion.expand_weekly_mods(
        spec,
        bd_only,
        expand_weekly_cached=expand_weekly_cached,
    )


@_ttl_lru_cache(maxsize=128)
def expand_yearly_cached(spec: str, y: int):
    return _cached_expansion.expand_yearly(
        spec,
        y,
        rewrite_month_names_to_ranges=_rewrite_month_names_to_ranges,
        split_csv_lower=_split_csv_lower,
        re_mod=re,
        month_len=month_len,
        yearfmt=_yearfmt,
    )



@_ttl_lru_cache(maxsize=128)
def expand_monthly_cached(spec: str, y: int, m: int):
    return _cached_expansion.expand_monthly(
        spec,
        y,
        m,
        month_len=month_len,
        expand_monthly_aliases=_expand_monthly_aliases,
        split_csv_lower=_split_csv_lower,
        nth_weekday_re=_nth_weekday_re,
        bd_re=_bd_re,
        weekday_map=_WEEKDAYS,
        re_mod=re,
    )


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
    return _schedule_utils.roll_apply(dt, mods, parse_error_cls=ParseError)

def _weeks_between(d1: date, d2: date) -> int:
    return _schedule_utils.weeks_between(d1, d2)

def apply_day_offset(d: date, mods: dict) -> date:
    return _schedule_utils.apply_day_offset(d, mods)


def base_next_after_atom(atom, ref_d: date) -> date:
    return _scheduler_atom.base_next_after_atom(
        atom,
        ref_d,
        expand_weekly_cached_mods=expand_weekly_cached_mods,
        split_csv_tokens=_split_csv_tokens,
        expand_monthly_cached=expand_monthly_cached,
        expand_yearly_cached=expand_yearly_cached,
        weekly_rand_pick=_weekly_rand_pick,
        week_monday=_week_monday,
        date_cls=date,
    )


# ------------------------------------------------------------------------------
# Anchor scheduling & iteration helpers
# ------------------------------------------------------------------------------
def _interval_allowed_for_atom(typ: str, ival: int, seed: date, cand: date) -> bool:
    return _scheduler_atom.interval_allowed_for_atom(
        typ,
        ival,
        seed,
        cand,
        weeks_between=_weeks_between,
        year_index=_year_index,
    )


def _advance_probe_for_interval_bucket(typ: str, ival: int, seed: date, cand: date) -> date:
    return _scheduler_atom.advance_probe_for_interval_bucket(
        typ,
        ival,
        seed,
        cand,
        weeks_between=_weeks_between,
        year_index=_year_index,
        date_cls=date,
    )


def _month_doms_safe(spec: str, y: int, m: int) -> list[int]:
    return _monthly_support.month_doms_safe(
        spec,
        y,
        m,
        expand_monthly_cached=expand_monthly_cached,
    )


def _month_has_hit(spec: str, y: int, m: int) -> bool:
    return _monthly_support.month_has_hit(
        spec,
        y,
        m,
        month_doms_safe=_month_doms_safe,
    )


def _first_hit_after_probe_in_month(spec: str, y: int, m: int, probe: date) -> date | None:
    return _monthly_support.first_hit_after_probe_in_month(
        spec,
        y,
        m,
        probe,
        month_doms_safe=_month_doms_safe,
    )


def _next_valid_month_on_or_after(spec: str, y: int, m: int) -> tuple[int, int]:
    return _monthly_support.next_valid_month_on_or_after(
        spec,
        y,
        m,
        month_has_hit=_month_has_hit,
    )


def _advance_k_valid_months(spec: str, start_y: int, start_m: int, k: int) -> tuple[int, int]:
    return _monthly_support.advance_k_valid_months(
        spec,
        start_y,
        start_m,
        k,
        next_valid_month_on_or_after=_next_valid_month_on_or_after,
    )


def _monthly_align_base_for_interval(spec: str, base: date, probe: date, seed: date, ival: int) -> date:
    return _monthly_support.monthly_align_base_for_interval(
        spec,
        base,
        probe,
        seed,
        ival,
        month_has_hit=_month_has_hit,
        next_valid_month_on_or_after=_next_valid_month_on_or_after,
        first_hit_after_probe_in_month=_first_hit_after_probe_in_month,
        advance_k_valid_months=_advance_k_valid_months,
        month_doms_safe=_month_doms_safe,
    )


def _accept_roll_candidate(ref_d: date, base: date, cand: date, roll_kind: str | None) -> bool:
    return _scheduler_atom.accept_roll_candidate(ref_d, base, cand, roll_kind)


def next_after_atom_with_mods(atom, ref_d: date, default_seed: date) -> date:
    return _scheduler_atom.next_after_atom_with_mods(
        atom,
        ref_d,
        default_seed,
        active_mod_keys=_active_mod_keys,
        base_next_after_atom=base_next_after_atom,
        interval_allowed_for_atom=_interval_allowed_for_atom,
        advance_probe_for_interval_bucket=_advance_probe_for_interval_bucket,
        monthly_align_base_for_interval=_monthly_align_base_for_interval,
        roll_apply=roll_apply,
        apply_day_offset=apply_day_offset,
        accept_roll_candidate=_accept_roll_candidate,
        max_anchor_iter=MAX_ANCHOR_ITER,
        warn_once_per_day=_warn_once_per_day,
        os_mod=os,
    )




def atom_matches_on(atom, d: date, default_seed: date) -> bool:
    return _scheduler_atom.atom_matches_on(
        atom,
        d,
        default_seed,
        next_after_atom_with_mods=next_after_atom_with_mods,
    )


def next_after_term(term, ref_d: date, default_seed: date):
    return _scheduler_expr.next_after_term(
        term,
        ref_d,
        default_seed,
        next_after_atom_with_mods=next_after_atom_with_mods,
        atom_matches_on=atom_matches_on,
        intersection_guard_steps=INTERSECTION_GUARD_STEPS,
    )


def next_after_expr(dnf, after_date, default_seed=None, seed_base=None):
    return _scheduler_expr.next_after_expr(
        dnf,
        after_date,
        default_seed=default_seed,
        seed_base=seed_base,
        active_mod_keys=_active_mod_keys,
        expand_weekly_cached=expand_weekly_cached,
        term_rand_info=_term_rand_info,
        atype=_atype,
        next_for_and=_next_for_and,
        months_since=_months_since,
        term_candidates_in_month=_term_candidates_in_month,
        sha_pick=_sha_pick,
        next_after_term=next_after_term,
    )


def anchors_between_expr(dnf, start_excl, end_excl, default_seed, seed_base=None):
    return _precompute.anchors_between_expr(
        dnf,
        start_excl,
        end_excl,
        default_seed,
        seed_base=seed_base,
        until_count_cap=UNTIL_COUNT_CAP,
        next_after_expr=next_after_expr,
        anchors_between_large_range=_anchors_between_large_range,
        warn_once_per_day=_warn_once_per_day,
        os_mod=os,
    )


def _anchors_between_large_range(
    dnf, start_excl, end_excl, default_seed, seed_base=None
):
    return _precompute.anchors_between_large_range(
        dnf,
        start_excl,
        end_excl,
        default_seed,
        seed_base=seed_base,
        until_count_cap=UNTIL_COUNT_CAP,
        next_after_expr=next_after_expr,
    )


def expr_has_m_or_y(dnf) -> bool:
    return _schedule_utils.expr_has_m_or_y(dnf)


def pick_hhmm_from_dnf_for_date(dnf, target: date, default_seed: date):
    return _schedule_utils.pick_hhmm_from_dnf_for_date(
        dnf,
        target,
        default_seed,
        atom_matches_on=atom_matches_on,
    )


# ------------------------------------------------------------------------------
# Datetime construction (local wall-clock -> UTC)
# ------------------------------------------------------------------------------
def build_local_datetime(d: date, hhmm=(DEFAULT_DUE_HOUR, 0)) -> datetime:
    return _timeutil.build_local_datetime(d, hhmm, _LOCAL_TZ)



# ------------------------------------------------------------------------------
# Yearly token helpers
# ------------------------------------------------------------------------------
def _iter_y_segments(s: str):
    yield from _linting.iter_y_segments(s, re_mod=re)


def _lint_expand_year_month_aliases(s: str) -> str:
    return _linting.lint_expand_year_month_aliases(
        s,
        month_from_alias=_month_from_alias,
        year_full_month_range_token=_year_full_month_range_token,
        re_mod=re,
    )


def _lint_check_weekly_delimiter_contract(s: str) -> str | None:
    return _linting.lint_check_weekly_delimiter_contract(s, re_mod=re)


def _lint_check_yearly_segments(s: str) -> str | None:
    return _linting.lint_check_yearly_segments(
        s,
        yearfmt=_yearfmt,
        iter_y_segments=_iter_y_segments,
        split_csv_tokens=_split_csv_tokens,
        re_mod=re,
    )


def _lint_check_global_md_dm_confusion(s: str) -> str | None:
    return _linting.lint_check_global_md_dm_confusion(
        s,
        yearfmt=_yearfmt,
        re_mod=re,
    )


def _lint_check_invalid_weekday_names(s: str) -> str | None:
    return _linting.lint_check_invalid_weekday_names(
        s,
        wd_abbr=_WD_ABBR,
        re_mod=re,
        difflib_mod=difflib,
    )


def _lint_check_nth_weekday_suffixes(s: str) -> str | None:
    return _linting.lint_check_nth_weekday_suffixes(s, re_mod=re)


def _lint_check_unsat_pure_weekly_and(s: str) -> str | None:
    return _linting.lint_check_unsat_pure_weekly_and(
        s,
        wd_abbr=_WD_ABBR,
        split_csv_tokens=_split_csv_tokens,
        re_mod=re,
    )


def _lint_check_backward_quarter_ranges(s: str) -> str | None:
    return _linting.lint_check_backward_quarter_ranges(s, re_mod=re)


def _lint_collect_warnings(s: str) -> list[str]:
    return _linting.lint_collect_warnings(s, re_mod=re)


def lint_anchor_expr(expr: str) -> tuple[str | None, list[str]]:
    return _linting.lint_anchor_expr(
        expr,
        unwrap_quotes=_unwrap_quotes,
        lint_expand_year_month_aliases=_lint_expand_year_month_aliases,
        lint_check_weekly_delimiter_contract=_lint_check_weekly_delimiter_contract,
        lint_check_yearly_segments=_lint_check_yearly_segments,
        lint_check_global_md_dm_confusion=_lint_check_global_md_dm_confusion,
        lint_check_invalid_weekday_names=_lint_check_invalid_weekday_names,
        lint_check_nth_weekday_suffixes=_lint_check_nth_weekday_suffixes,
        lint_check_unsat_pure_weekly_and=_lint_check_unsat_pure_weekly_and,
        lint_check_backward_quarter_ranges=_lint_check_backward_quarter_ranges,
        lint_collect_warnings=_lint_collect_warnings,
        re_mod=re,
    )



def _rewrite_weekly_multi_time_atoms(s: str) -> str:
    return _parser_frontend.rewrite_weekly_multi_time_atoms(
        s,
        split_csv_tokens=_split_csv_tokens,
        re_mod=re,
    )

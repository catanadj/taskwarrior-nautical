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
from typing import Any, Callable, TYPE_CHECKING, TypeAlias, TypedDict, cast
from datetime import datetime, timedelta, timezone, date
from functools import lru_cache, partial
from calendar import month_name, monthrange
from datetime import date as _date

if TYPE_CHECKING:
    from .parser_models import AnchorDNF as AnchorDNFType
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

_parser_models = importlib.import_module(f"{_PKG_IMPORT_ROOT}.parser_models")
_scheduler_models = importlib.import_module(f"{_PKG_IMPORT_ROOT}.scheduler_models")
AnchorMods = _parser_models.AnchorMods
AnchorAtom = _parser_models.AnchorAtom
AnchorTerm = _parser_models.AnchorTerm
AnchorDNF = _parser_models.AnchorDNF
AnchorValidationResult = _parser_models.AnchorValidationResult
ParseError = _parser_models.ParseError
YearTokenFormatError = _parser_models.YearTokenFormatError
AndTermUnsatisfiable = _parser_models.AndTermUnsatisfiable
OccurrenceSearchExhausted = _scheduler_models.OccurrenceSearchExhausted


def _import_sibling(module_name: str):
    _PKG_PROXY.__dict__.update(globals())
    return importlib.import_module(f"{_PKG_IMPORT_ROOT}.{module_name}")


class _LazySibling:
    """Resolve a focused sibling module only when one of its APIs is used."""

    __slots__ = ("_name", "_module")

    def __init__(self, module_name: str):
        self._name = module_name
        self._module = None

    def _resolve(self):
        if self._module is None:
            self._module = _import_sibling(self._name)
        return self._module

    def __getattr__(self, name: str):
        return getattr(self._resolve(), name)


class _LazyApiBundle:
    """Resolve one core-bound API bundle on its first facade call.

    API modules are intentionally imported only when their public aliases are
    used.  The aliases remain ordinary callables after resolution, preserving
    the existing facade namespace and monkeypatch points.
    """

    __slots__ = ("_module_name", "_module", "_core", "_namespace", "_aliases", "_bindings")

    def __init__(self, module_name: str, aliases: tuple[str, ...], *, core: Any, namespace: dict[str, Any]):
        self._module_name = module_name
        self._module = None
        self._core = core
        self._namespace = namespace
        self._aliases = aliases
        self._bindings = None

    def _resolve(self):
        if self._bindings is None:
            module = _import_sibling(self._module_name)
            self._module = module
            self._bindings = module.for_core(self._core, namespace=self._namespace)
            for spec in self._aliases:
                alias, source = spec if isinstance(spec, tuple) else (spec, spec)
                self._namespace[alias] = getattr(self._bindings, source)
        return self._bindings

    def alias(self, name: str, source_name: str | None = None):
        source_name = source_name or name

        def call(*args, **kwargs):
            return getattr(self._resolve(), source_name)(*args, **kwargs)

        call.__name__ = name
        call.__qualname__ = name
        return call

    def __getattr__(self, name: str):
        return getattr(self._resolve(), name)


TaskDict: TypeAlias = dict[str, Any]


class HintMetaCfg(TypedDict, total=False):
    fmt: str
    salt: str
    tz: str
    hol: str
    bc: str


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
    dnf: AnchorDNFType
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


# --- Core constants ---
_CACHE_LOCK_RETRIES = 6
_CACHE_LOCK_SLEEP_BASE = 0.05
_CACHE_LOCK_JITTER = 0.0
_CACHE_LOCK_STALE_AFTER = 300.0
_CACHE_LOAD_MEM_MAX = 128
_CACHE_LOAD_MEM_TTL = 300
_CACHE_LOAD_MEM: OrderedDict[str, tuple[tuple[int, int, int, int], dict, float]] = OrderedDict()
_core_config = _import_sibling("core_config")
_env_flag_true = _core_config.env_flag_true
_path_input_error = _core_config.path_input_error
_normalized_abspath = _core_config.normalized_abspath
_nearest_existing_dir = _core_config.nearest_existing_dir
_world_writable_without_sticky = _core_config.world_writable_without_sticky
_path_safety_error = _core_config.path_safety_error
_validated_user_dir = _core_config.validated_user_dir
_DEFAULTS = _core_config._DEFAULTS
_read_toml = _core_config._read_toml
_config_paths = _core_config._config_paths
_warn_env_config_missing = _core_config._warn_env_config_missing
_normalize_keys = _core_config._normalize_keys
_load_config = _core_config._load_config
_nautical_cache_dir = _core_config.nautical_cache_dir
_warn_once_per_day = _core_config.warn_once_per_day
_warn_once_per_day_any = _core_config.warn_once_per_day_any
_warn_rate_limited_any = _core_config.warn_rate_limited_any
_warn_toml_parse_error = _core_config._warn_toml_parse_error
_get_config = _core_config._get_config
effective_config_snapshot = _core_config.effective_config_snapshot
effective_config_fingerprint = _core_config.effective_config_fingerprint
scheduler_config_fingerprint = _core_config.scheduler_config_fingerprint
configuration_drift = _core_config.configuration_drift
_CONF = _core_config._CONF
_conf_raw = _core_config.conf_raw
_conf_str = _core_config.conf_str
_conf_int = _core_config.conf_int
_conf_bool = _core_config.conf_bool
_conf_csv_or_list = _core_config.conf_csv_or_list
_conf_uda_field_list = _core_config.conf_uda_field_list
_trueish = _core_config.trueish
_ttl_lru_cache = _core_config.ttl_lru_cache


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
    try:
        _position_selection.clear_candidate_cache()
        _selection_inner_matcher.cache_clear()
    except Exception:
        pass


# -------- UI helpers ----------------------------------------------------------
_ui = _LazySibling("ui")


def _lazy_ui_call(name: str):
    def call(*args, **kwargs):
        return getattr(_ui, name)(*args, **kwargs)

    call.__name__ = name
    return call


strip_rich_markup = _lazy_ui_call("strip_rich_markup")
term_width_stderr = _lazy_ui_call("term_width_stderr")
fast_color_enabled = _lazy_ui_call("fast_color_enabled")
ansi = _lazy_ui_call("ansi")
emit_wrapped = _lazy_ui_call("emit_wrapped")
emit_line = _lazy_ui_call("emit_line")
text_line = _lazy_ui_call("text_line")
panel_line_from_rows = _lazy_ui_call("panel_line_from_rows")
panel_line = _lazy_ui_call("panel_line")
panel_themes = _lazy_ui_call("panel_themes")

chain_colour_root = _import_sibling("panel_colours").chain_colour_root
DiagnosticEvent = _import_sibling("diagnostic_models").DiagnosticEvent


def render_panel(*args, **kwargs):
    ui = _ui._resolve()
    ui.panel_line = panel_line
    ui.text_line = text_line
    return ui.render_panel(*args, **kwargs)

ANCHOR_YEAR_FMT = _core_config.ANCHOR_YEAR_FMT
WRAND_SALT = _core_config.WRAND_SALT
LOCAL_TZ_NAME = _core_config.LOCAL_TZ_NAME
CONFIG_ERROR = _core_config.configuration_error()
SEASON_HEMISPHERE = _core_config.SEASON_HEMISPHERE
HOLIDAY_REGION = _core_config.HOLIDAY_REGION
ANCHOR_FILE_DIR = _core_config.ANCHOR_FILE_DIR
OMIT_FILE_DIR = _core_config.OMIT_FILE_DIR
ANCHOR_PRESETS = _core_config.ANCHOR_PRESETS
OMIT_PRESETS = _core_config.OMIT_PRESETS
BUSINESS_CALENDAR_CONFIG = _core_config.BUSINESS_CALENDAR_CONFIG
ASTRONOMY_CONFIG = _core_config.ASTRONOMY_CONFIG
ENABLE_ANCHOR_CACHE = _core_config.ENABLE_ANCHOR_CACHE
ENABLE_UDA_ALIASES = _core_config.ENABLE_UDA_ALIASES
ANCHOR_CACHE_DIR_OVERRIDE = _core_config.ANCHOR_CACHE_DIR_OVERRIDE
ANCHOR_CACHE_TTL = _core_config.ANCHOR_CACHE_TTL
CHAIN_COLOR_PER_CHAIN = _core_config.CHAIN_COLOR_PER_CHAIN
SHOW_TIMELINE_GAPS = _core_config.SHOW_TIMELINE_GAPS
SHOW_ANALYTICS = _core_config.SHOW_ANALYTICS
ANALYTICS_STYLE = _core_config.ANALYTICS_STYLE
ANALYTICS_ONTIME_TOL_SECS = _core_config.ANALYTICS_ONTIME_TOL_SECS
DEBUG_WAIT_SCHED = _core_config.DEBUG_WAIT_SCHED
CHECK_CHAIN_INTEGRITY = _core_config.CHECK_CHAIN_INTEGRITY
PANEL_MODE = _core_config.PANEL_MODE
LIVE_PANEL_DURATION_MS = _core_config.LIVE_PANEL_DURATION_MS
LIVE_PANEL_FOOTER = _core_config.LIVE_PANEL_FOOTER
FAST_COLOR = _core_config.FAST_COLOR
EXIT_PROGRESS = _core_config.EXIT_PROGRESS
SPAWN_QUEUE_MAX_BYTES = _core_config.SPAWN_QUEUE_MAX_BYTES
SPAWN_QUEUE_DRAIN_MAX_ITEMS = _core_config.SPAWN_QUEUE_DRAIN_MAX_ITEMS
MAX_CHAIN_WALK = _core_config.MAX_CHAIN_WALK
MAX_ANCHOR_ITER = _core_config.MAX_ANCHOR_ITER
MAX_LINK_NUMBER = _core_config.MAX_LINK_NUMBER
SANITIZE_UDA = _core_config.SANITIZE_UDA
SANITIZE_UDA_MAX_LEN = _core_config.SANITIZE_UDA_MAX_LEN
MAX_JSON_BYTES = _core_config.MAX_JSON_BYTES
RECURRENCE_UPDATE_UDAS = _core_config.RECURRENCE_UPDATE_UDAS
_CACHE_TTL_SECS = _core_config.CACHE_TTL_SECS
_CACHE_LOAD_MEM_MAX = _core_config.CACHE_LOAD_MEM_MAX
_CACHE_LOAD_MEM_TTL = _core_config.CACHE_LOAD_MEM_TTL

# ==============================================================================
# SECTION: Taskwarrior helpers
# ==============================================================================
_common = _import_sibling("common")
_cache_payload = _import_sibling("cache_payload")
_cache_locking = _import_sibling("cache_locking")
_acf_support = _import_sibling("acf_support")
_business_calendar = _import_sibling("business_calendar")
_business_calendar_config = _import_sibling("business_calendar_config")
_cached_expansion = _LazySibling("cached_expansion")
_nth_monthly = _LazySibling("nth_monthly")
_expansion_support = _LazySibling("expansion_support")
_monthly_support = _LazySibling("monthly_support")
_natural_language = _LazySibling("natural_language")
_astronomy = _LazySibling("astronomy")
_linting = _LazySibling("linting")
_parser_atoms = _LazySibling("parser_atoms")
_parser_dnf = _LazySibling("parser_dnf")
_parser_frontend = _LazySibling("parser_frontend")
_position_selection = _LazySibling("position_selection")
_season_support = _import_sibling("season_support")
_season_support.configure_hemisphere(SEASON_HEMISPHERE)
_precompute = _LazySibling("precompute")
_quarter_helpers = _LazySibling("quarter_helpers")
_quarter_rewrite = _LazySibling("quarter_rewrite")
_quarter_selector = _LazySibling("quarter_selector")
_satisfiability = _LazySibling("satisfiability")
_schedule_utils = _LazySibling("schedule_utils")
_scheduler_atom = _LazySibling("scheduler_atom")
_scheduler_expr = _LazySibling("scheduler_expr")
_strict_validation = _LazySibling("strict_validation")
_tokenutil = _import_sibling("tokenutil")
_yearly_parse = _LazySibling("yearly_parse")
_yearly_validation = _LazySibling("yearly_validation")
_year_tokens = _LazySibling("year_tokens")

short_uuid = _common.short_uuid
DEFAULT_BUSINESS_CALENDAR = _business_calendar.DEFAULT_BUSINESS_CALENDAR
BusinessCalendarConfigError = _business_calendar_config.BusinessCalendarConfigError
business_calendar_displacement_for_date = _business_calendar.business_calendar_displacement_for_date
capture_business_calendar_displacements = _business_calendar.capture_business_calendar_displacements


def _with_business_calendar(fn: Callable[..., Any], business_calendar) -> Callable[..., Any]:
    business_calendar = _business_calendar.effective_business_calendar(business_calendar)
    if business_calendar is DEFAULT_BUSINESS_CALENDAR:
        return fn
    return partial(fn, business_calendar=business_calendar)

# ==============================================================================
# SECTION: Time & timezone helpers
# ==============================================================================
try:
    import zoneinfo as _zoneinfo
except Exception:
    _zoneinfo = None

_LOCAL_TZ: Any = None
_TIMEZONE_CONFIG_ERROR = ""


def _refresh_timezone() -> None:
    global _LOCAL_TZ, _TIMEZONE_CONFIG_ERROR
    if _zoneinfo is None:
        _TIMEZONE_CONFIG_ERROR = "timezone support unavailable (zoneinfo import failed)"
        _LOCAL_TZ = None
        _warn_once_per_day(
            "timezone_zoneinfo_unavailable",
            "[nautical] timezone support unavailable (zoneinfo import failed); using UTC fallback.",
        )
        return
    try:
        _LOCAL_TZ = _zoneinfo.ZoneInfo(LOCAL_TZ_NAME)
        _TIMEZONE_CONFIG_ERROR = ""
    except Exception:
        _TIMEZONE_CONFIG_ERROR = f"configured timezone '{LOCAL_TZ_NAME}' is invalid or unavailable"
        _LOCAL_TZ = None
        _warn_once_per_day(
            "timezone_local_invalid",
            f"[nautical] timezone '{LOCAL_TZ_NAME}' is invalid/unavailable; using UTC fallback.",
        )


_refresh_timezone()

_timeutil = _import_sibling("timeutil")


def scheduling_configuration_error() -> str:
    """Return a blocking configuration error for Nautical scheduling paths."""
    if CONFIG_ERROR:
        return CONFIG_ERROR
    return _TIMEZONE_CONFIG_ERROR


def reload_taskdata_config(taskdata: str | os.PathLike[str]) -> dict[str, str | bool]:
    """Apply the validated configuration selected for a Taskwarrior data directory."""
    global CONFIG_ERROR
    result = _core_config.reload_for_taskdata(taskdata)
    if not result.get("ok"):
        error = str(result.get("error") or "configuration unavailable")
        CONFIG_ERROR = f"Nautical configuration reload failed: {error}"
        raise RuntimeError(f"Nautical configuration reload failed: {error}")

    for name in (
        "WRAND_SALT",
        "LOCAL_TZ_NAME",
        "SEASON_HEMISPHERE",
        "HOLIDAY_REGION",
        "ANCHOR_FILE_DIR",
        "OMIT_FILE_DIR",
        "ANCHOR_PRESETS",
        "OMIT_PRESETS",
        "BUSINESS_CALENDAR_CONFIG",
        "ASTRONOMY_CONFIG",
        "ENABLE_ANCHOR_CACHE",
        "ENABLE_UDA_ALIASES",
        "ANCHOR_CACHE_DIR_OVERRIDE",
        "ANCHOR_CACHE_TTL",
        "CHAIN_COLOR_PER_CHAIN",
        "SHOW_TIMELINE_GAPS",
        "SHOW_ANALYTICS",
        "ANALYTICS_STYLE",
        "ANALYTICS_ONTIME_TOL_SECS",
        "DEBUG_WAIT_SCHED",
        "CHECK_CHAIN_INTEGRITY",
        "PANEL_MODE",
        "LIVE_PANEL_DURATION_MS",
        "LIVE_PANEL_FOOTER",
        "FAST_COLOR",
        "EXIT_PROGRESS",
        "SPAWN_QUEUE_MAX_BYTES",
        "SPAWN_QUEUE_DRAIN_MAX_ITEMS",
        "MAX_CHAIN_WALK",
        "MAX_ANCHOR_ITER",
        "MAX_LINK_NUMBER",
        "SANITIZE_UDA",
        "SANITIZE_UDA_MAX_LEN",
        "MAX_JSON_BYTES",
        "RECURRENCE_UPDATE_UDAS",
        "_CACHE_TTL_SECS",
        "_CACHE_LOAD_MEM_MAX",
        "_CACHE_LOAD_MEM_TTL",
    ):
        globals()[name] = getattr(_core_config, name if not name.startswith("_") else name[1:])
    globals()["_CONF"] = _core_config._CONF
    CONFIG_ERROR = _core_config.configuration_error()
    _refresh_timezone()
    _season_support.configure_hemisphere(SEASON_HEMISPHERE)
    error = scheduling_configuration_error()
    if error:
        raise RuntimeError(f"Invalid Nautical configuration: {error}")
    return result


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


# -------- Pre-compiled Regex Patterns ----------
_int_floatish_re = _common._INT_FLOATISH_RE
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

_token_api = _import_sibling("token_api").for_core(
    sys.modules[__name__],
    namespace=globals(),
)
_yearfmt = _token_api._yearfmt
_tok = _token_api._tok
_tok_range = _token_api._tok_range
_safe_match = _token_api._safe_match
sanitize_text = _token_api.sanitize_text
sanitize_task_strings = _token_api.sanitize_task_strings
_split_csv_tokens = _token_api._split_csv_tokens
_split_csv_lower = _token_api._split_csv_lower
_iso_week_index = _token_api._iso_week_index
_month_index = _token_api._month_index
_year_index = _token_api._year_index
_static_month_last_day = _token_api._static_month_last_day
_month_from_alias = _token_api._month_from_alias
_year_full_months_span_token = _token_api._year_full_months_span_token
_rewrite_month_names_to_ranges = _token_api._rewrite_month_names_to_ranges
_unwrap_quotes = _token_api._unwrap_quotes
_year_full_month_range_token = _token_api._year_full_month_range_token
_mon_to_int = _token_api._mon_to_int
_expand_weekly_aliases = _token_api._expand_weekly_aliases
_expand_monthly_aliases = _token_api._expand_monthly_aliases
_normalize_weekday = _token_api._normalize_weekday

_ACF_COMPRESSED = True
ACF_COMPRESSED = _ACF_COMPRESSED
ACF_CHECKSUM_LEN = 8
_WD_ABBR = _tokenutil.WD_ABBR
_WEEKLY_ALIAS = _tokenutil.WEEKLY_ALIAS
_MONTHLY_ALIAS = _tokenutil.MONTHLY_ALIAS

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


# ---- Core iterator over DNF ---------------------------------------------------
_NTH_RE  = re.compile(r"^(?:(\d)(?:st|nd|rd|th)|last)-(" + "|".join(_WD_ABBR) + r")$")

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


# Keep this parser guard local so the quarter selector module remains lazy.
_MONTH_SELECTOR_MAX_LEN = 64


_dates = _import_sibling("dates")


_recurrence_metadata = _import_sibling("recurrence_metadata")
_active_mod_keys = _recurrence_metadata.active_mod_keys
_atype = _recurrence_metadata.atom_type
_aspec = _recurrence_metadata.atom_spec
_amods = _recurrence_metadata.atom_mods
_ainterval = _recurrence_metadata.atom_interval


# Shared weekday indices remain part of the facade namespace for scheduler
# callbacks and third-party callers.
_WD = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}


_months_since = _recurrence_metadata.months_since


# CP parsing is implemented in a focused module; these aliases preserve the
# long-standing ``nautical_core`` facade used by hooks and third-party callers.
_cp_parser = _import_sibling("cp_parser")
parse_cp_duration = _cp_parser.parse_cp_duration
parse_cp_sequence_tokens = _cp_parser.parse_cp_sequence_tokens
parse_cp_sequence = _cp_parser.parse_cp_sequence
cp_sequence_parse_error = _cp_parser.cp_sequence_parse_error
cp_sequence_interval_for_token = _cp_parser.cp_sequence_interval_for_token
cp_sequence_interval_for_link = _cp_parser.cp_sequence_interval_for_link


_recurrence_candidates = _import_sibling("recurrence_candidates").for_core(
    sys.modules[__name__],
    namespace=globals(),
)
_anchors_between_large_range = _recurrence_candidates.anchors_between_large_range
anchors_between_expr = _recurrence_candidates.anchors_between_expr


# Quarter rewrite entry points are bound before parser construction because
# the parser's DNF pipeline applies quarter normalization.
_quarter_api = _LazyApiBundle(
    "quarter_api",
    (
        "_yearly_tokens",
        "_monthly_tokens",
        "_quarters_from_first_month_tokens",
        "_quarters_from_start_day_tokens",
        "_quarters_from_end_day_tokens",
        "_format_quarter_set",
        "_rewrite_quarter_spec_mode",
        "_quarter_atom_spec",
        "_has_quarter_tokens",
        "_has_plain_quarter_tokens",
        "_is_start_month_selector",
        "_is_end_month_selector",
        "_quarter_month_selector_mode",
        "_term_quarter_rewrite_mode",
        "_rewrite_quarter_year_atoms",
        "_rewrite_quarters_in_context",
    ),
    core=sys.modules[__name__],
    namespace=globals(),
)
for _name in _quarter_api._aliases:
    globals()[_name] = _quarter_api.alias(_name)

# ACF entry points are bound before parser construction because parser
# validation and canonical-form generation share these compatibility names.
_acf_api = _LazyApiBundle(
    "acf_api",
    (
        "_atom_sort_key",
        "_acf_unpack",
        "_year_pair_cached",
        "_year_pair",
        "_normalize_spec_for_acf_uncached",
        "_normalize_spec_for_acf_cached",
        "_normalize_spec_for_acf",
        "_mods_to_acf",
        "_acf_mods_to_string",
        "_acf_spec_to_string",
        "_build_acf_impl",
        "is_valid_acf",
        "acf_to_original_format",
    ),
    core=sys.modules[__name__],
    namespace=globals(),
)
for _name in _acf_api._aliases:
    globals()[_name] = _acf_api.alias(_name)

_expansion_api = _LazyApiBundle(
    "expansion_api",
    (
        "_days_in_month",
        "_wd_idx",
        "_wday_idx_any",
        "_weekly_spec_to_wset",
        "_doms_for_weekly_spec",
        "_doms_for_monthly_token",
        "_y_ranges_from_spec",
        "_doms_allowed_by_year",
        "_month_allowed_doms_for_monthly_atom",
        "_intersect_monthly_atoms_allowed",
    ),
    core=sys.modules[__name__],
    namespace=globals(),
)
for _name in _expansion_api._aliases:
    globals()[_name] = _expansion_api.alias(_name)

_parser_support_api = _LazyApiBundle(
    "parser_support_api",
    (
        "_parse_hhmm",
        "_parse_atom_head",
        "_parse_atom_mods",
        "_parse_y_token_cached",
        "_parse_y_token",
        "_rewrite_year_month_aliases_in_context",
        "_fatal_bad_colon_in_year_tail",
        "_raise_on_bad_colon_year_tokens",
        "_skip_ws_pos",
        "_raise_if_comma_joined_anchors",
        "_parse_anchor_expr_to_dnf_cached_obj",
        "_parse_anchor_expr_to_dnf_cached_impl",
        "_validate_weekly_spec",
        "_validate_monthly_spec",
        "_split_inline_items_respecting_t_lists",
        "_parse_group_with_inline_mods",
        "_rewrite_weekly_multi_time_atoms",
    ),
    core=sys.modules[__name__],
    namespace=globals(),
)
for _name in _parser_support_api._aliases:
    globals()[_name] = _parser_support_api.alias(_name)

# Parser entry points live in ``parser_api``; retain these aliases for the
# established ``nautical_core`` import contract.
_parser_api = _LazyApiBundle(
    "parser_api",
    (
        "build_acf",
        "resolve_anchor_presets",
        "resolve_omit_presets",
        "anchor_preset_display",
        "omit_preset_display",
        "_resolve_preset_refs",
        "_resolve_anchor_presets_impl",
        "_normalize_anchor_expr_input",
        "_normalize_monthly_ordinal_spec",
        "_build_anchor_atom_dnf",
        "_parse_anchor_atom_at",
        "_yearly_pair_from_fmt",
        "_yearly_mmdd_error",
        "_validate_yearly_token_allowlist",
        "_validate_yearly_token_detailed",
        "_validate_yearly_token_format",
        "_validate_year_tokens_in_dnf",
        "_validate_yearly_token",
        "_yearly_last_day",
        "_yearly_check_day_month",
        "_validate_yearly_spec_token",
        "_validate_yearly_spec",
        "_weekday_set_from_weekly_atom",
        "_md_pairs_from_yearly_spec",
        "_quick_weekly_and_check",
        "_quick_yearly_and_check",
        "_quick_moon_and_check",
        "_term_has_any_match_within",
        "_validate_and_terms_satisfiable",
        "parse_anchor_expr_to_dnf",
        "parse_anchor_expr_to_dnf_cached",
        "validate_anchor_expr_strict",
        ("_normalize_anchor_input_to_dnf", "normalize_anchor_input_to_dnf"),
        ("_assert_dnf_structure_strict", "assert_dnf_structure_strict"),
        ("_validate_anchor_atom_strict", "validate_anchor_atom_strict"),
        ("_validate_anchor_dnf_atoms_strict", "validate_anchor_dnf_atoms_strict"),
    ),
    core=sys.modules[__name__],
    namespace=globals(),
)
for _name in _parser_api._aliases:
    _alias_name, _source_name = _name if isinstance(_name, tuple) else (_name, _name)
    globals()[_alias_name] = _parser_api.alias(_alias_name, _source_name)

# Business-calendar configuration is bound after parser APIs are available so
# calendar rules can reuse the same strict anchor/omit validators.
_business_calendar_api = _import_sibling("business_calendar_api").for_core(
    sys.modules[__name__],
    namespace=globals(),
)
business_calendar_definitions = _business_calendar_api.business_calendar_definitions
_validate_business_calendar_omit_expr = _business_calendar_api._validate_business_calendar_omit_expr
_business_calendar_expression_matches_date = _business_calendar_api._business_calendar_expression_matches_date
resolve_business_calendar_config = _business_calendar_api.resolve_business_calendar_config
configured_business_calendars = _business_calendar_api.configured_business_calendars
get_configured_business_calendar = _business_calendar_api.get_configured_business_calendar
business_calendar_for_task = _business_calendar_api.business_calendar_for_task
normalize_task_business_calendar = _business_calendar_api.normalize_task_business_calendar
business_calendar_fingerprint = _business_calendar_api.business_calendar_fingerprint
use_business_calendar = _business_calendar_api.use_business_calendar
use_task_business_calendar = _business_calendar_api.use_task_business_calendar

_linting_api = _import_sibling("linting_api").for_core(
    sys.modules[__name__],
    namespace=globals(),
)
_iter_y_segments = _linting_api._iter_y_segments
_lint_expand_year_month_aliases = _linting_api._lint_expand_year_month_aliases
_lint_check_weekly_delimiter_contract = _linting_api._lint_check_weekly_delimiter_contract
_lint_check_yearly_segments = _linting_api._lint_check_yearly_segments
_lint_check_global_md_dm_confusion = _linting_api._lint_check_global_md_dm_confusion
_lint_check_invalid_weekday_names = _linting_api._lint_check_invalid_weekday_names
_lint_check_nth_weekday_suffixes = _linting_api._lint_check_nth_weekday_suffixes
_lint_check_unsat_pure_weekly_and = _linting_api._lint_check_unsat_pure_weekly_and
_lint_check_backward_quarter_ranges = _linting_api._lint_check_backward_quarter_ranges
_lint_collect_warnings = _linting_api._lint_collect_warnings
lint_anchor_expr = _linting_api.lint_anchor_expr

# Scheduler entry points are bound to this exact core instance for isolated
# hook/test loaders while preserving the long-standing facade names.
_scheduler_api = _LazyApiBundle(
    "scheduler_api",
    (
        "_expand_weekly_cached_impl",
        "_expand_weekly_cached_mods_impl",
        "_expand_yearly_cached_impl",
        "_expand_monthly_cached_impl",
        "_expand_monthly_for_month_impl",
        "_expand_weekly_impl",
        "_expand_yearly_for_year_strict_impl",
        "_roll_apply_impl",
        "_month_doms_safe",
        "_month_has_hit",
        "_first_hit_after_probe_in_month",
        "_next_valid_month_on_or_after",
        "_advance_k_valid_months",
        "_monthly_align_base_for_interval",
        "_selection_inner_matcher",
        "_apply_selection_date_modifiers",
        "_week_monday",
        "_weekly_rand_pick",
        "_is_bd",
        "_random_identity",
        "_random_pick_index",
        "_random_pick_indices",
        "_term_rand_info",
        "dnf_has_counted_random",
        "_filter_by_w",
        "_month_tokens_for_atom_cached",
        "_month_tokens_for_atom",
        "_term_candidates_in_month",
        "_next_for_and_rand_yearly",
        "_next_for_and_fast_path",
        "_next_for_and",
        "_next_for_or",
        "expand_weekly_cached",
        "expand_weekly_cached_mods",
        "expand_yearly_cached",
        "expand_monthly_cached",
        "expand_monthly_for_month",
        "expand_weekly",
        "expand_yearly_for_year_strict",
        "roll_apply",
        "apply_day_offset",
        "base_next_after_atom",
        ("_interval_allowed_for_atom", "interval_allowed_for_atom"),
        ("_advance_probe_for_interval_bucket", "advance_probe_for_interval_bucket"),
        ("_accept_roll_candidate", "accept_roll_candidate"),
        "next_after_atom_with_mods",
        "atom_matches_on",
        "next_after_factor",
        "factor_matches_on",
        "next_after_term",
        "next_after_expr",
        "_weeks_between",
        "_resolve_moon_phase_date",
        "_moon_phase_matches_date",
    ),
    core=sys.modules[__name__],
    namespace=globals(),
)
for _name in _scheduler_api._aliases:
    _alias_name, _source_name = _name if isinstance(_name, tuple) else (_name, _name)
    globals()[_alias_name] = _scheduler_api.alias(_alias_name, _source_name)

# Date/time adapters are bound after scheduler construction so date-specific
# slot selection can reuse the facade's business-calendar callback.
_time_api = _import_sibling("time_api").for_core(
    sys.modules[__name__],
    namespace=globals(),
)
now_utc = _time_api.now_utc
to_local = _time_api.to_local
utc_to_local_naive = _time_api.utc_to_local_naive
local_naive_to_utc = _time_api.local_naive_to_utc
fmt_dt_local = _time_api.fmt_dt_local
fmt_isoz = _time_api.fmt_isoz
_ensure_utc = _time_api._ensure_utc
coerce_int = _time_api.coerce_int
parse_dt_any = _time_api.parse_dt_any
month_len = _time_api.month_len
add_months = _time_api.add_months
months_days_between = _time_api.months_days_between
humanize_delta = _time_api.humanize_delta
expr_has_m_or_y = _time_api.expr_has_m_or_y
pick_hhmm_from_dnf_for_date = _time_api.pick_hhmm_from_dnf_for_date
build_local_datetime = _time_api.build_local_datetime

_natural_language_api = _import_sibling("natural_language_api").for_core(
    sys.modules[__name__],
    namespace=globals(),
)
_ordinal = _natural_language_api._ordinal
_term_collect_mods = _natural_language_api._term_collect_mods
_fmt_hhmm_for_term = _natural_language_api._fmt_hhmm_for_term
_fmt_weekdays_list = _natural_language_api._fmt_weekdays_list
_fmt_monthly_atom = _natural_language_api._fmt_monthly_atom
_fmt_md = _natural_language_api._fmt_md
_is_full_month = _natural_language_api._is_full_month
_fmt_yearly_atom = _natural_language_api._fmt_yearly_atom
_describe_monthly_tokens = _natural_language_api._describe_monthly_tokens
_describe_is_pure_nth_weekday_spec = _natural_language_api._describe_is_pure_nth_weekday_spec
_describe_is_pure_dom_spec = _natural_language_api._describe_is_pure_dom_spec
_describe_single_full_month_from_yearly_spec = _natural_language_api._describe_single_full_month_from_yearly_spec
_describe_term_roll_shift = _natural_language_api._describe_term_roll_shift
_describe_term_bd_filter = _natural_language_api._describe_term_bd_filter
_describe_roll_suffix = _natural_language_api._describe_roll_suffix
_describe_inject_schedule_suffixes = _natural_language_api._describe_inject_schedule_suffixes
_describe_anchor_term_collect = _natural_language_api._describe_anchor_term_collect
_describe_anchor_term_fused_month_year = _natural_language_api._describe_anchor_term_fused_month_year
_describe_anchor_term_interval_prefix = _natural_language_api._describe_anchor_term_interval_prefix
_describe_anchor_term_parts = _natural_language_api._describe_anchor_term_parts
describe_anchor_term = _natural_language_api.describe_anchor_term
_describe_anchor_expr_from_dnf = _natural_language_api._describe_anchor_expr_from_dnf
_describe_anchor_expr_impl = _natural_language_api._describe_anchor_expr_impl
_term_prevnext_wd = _natural_language_api._term_prevnext_wd
_inject_prevnext_phrase = _natural_language_api._inject_prevnext_phrase
_join_natural_or_terms = _natural_language_api._join_natural_or_terms
_longest_common_suffix = _natural_language_api._longest_common_suffix
_compress_or_terms_by_clause = _natural_language_api._compress_or_terms_by_clause
_describe_anchor_dnf_impl = _natural_language_api._describe_anchor_dnf_impl
_normalize_range_token = _natural_language_api._normalize_range_token
_rand_bucket_time_from_mods = _natural_language_api._rand_bucket_time_from_mods
_rand_bucket_merge_mods = _natural_language_api._rand_bucket_merge_mods
_rand_bucket_signature = _natural_language_api._rand_bucket_signature
_try_bucket_rand_monthly = _natural_language_api._try_bucket_rand_monthly
describe_anchor_expr = _natural_language_api.describe_anchor_expr
describe_anchor_dnf = _natural_language_api.describe_anchor_dnf

_cache_api = _import_sibling("cache_api").for_core(
    sys.modules[__name__],
    namespace=globals(),
)
_safe_lock_sleep_once = _cache_api._safe_lock_sleep_once
_safe_lock_ensure_parent = _cache_api._safe_lock_ensure_parent
_safe_lock_age = _cache_api._safe_lock_age
_safe_lock_stale_pid = _cache_api._safe_lock_stale_pid
_safe_lock_fcntl_context = _cache_api._safe_lock_fcntl_context
_safe_lock_excl_context = _cache_api._safe_lock_excl_context
safe_lock = _cache_api.safe_lock
_cache_lock = _cache_api._cache_lock
_is_atom_like = _cache_api._is_atom_like
_is_dnf_like = _cache_api._is_dnf_like
_clone_mod_value = _cache_api._clone_mod_value
_clone_mods = _cache_api._clone_mods
_clone_atom = _cache_api._clone_atom
_clone_dnf = _cache_api._clone_dnf
_clone_cache_payload = _cache_api._clone_cache_payload
_normalize_dnf_cached = _cache_api._normalize_dnf_cached
_cache_payload_shape_ok = _cache_api._cache_payload_shape_ok
_cache_atomic_replace = _cache_api._cache_atomic_replace
_cache_dir = _cache_api._cache_dir
_cache_key = _cache_api._cache_key
_cache_path = _cache_api._cache_path
_cache_lock_path = _cache_api._cache_lock_path
_quarantine_cache = _cache_api._quarantine_cache
_cache_load_impl = _cache_api._cache_load_impl
_cache_save_impl = _cache_api._cache_save_impl
_cache_gc_impl = _cache_api._cache_gc_impl
_cache_key_for_task_cached = _cache_api._cache_key_for_task_cached
_cache_key_for_task_impl = _cache_api._cache_key_for_task_impl
cache_load = _cache_api.cache_load
cache_save = _cache_api.cache_save
cache_gc = _cache_api.cache_gc
cache_key_for_task = _cache_api.cache_key_for_task
_dnf_cache_fingerprint = _cache_api._dnf_cache_fingerprint
_dnf_cache_key = _cache_api._dnf_cache_key
_dnf_cache_load = _cache_api._dnf_cache_load
_dnf_cache_save = _cache_api._dnf_cache_save

_precompute_api = _import_sibling("precompute_api").for_core(
    sys.modules[__name__],
    namespace=globals(),
)
precompute_hints = _precompute_api.precompute_hints
build_and_cache_hints = _precompute_api.build_and_cache_hints

RecurrenceModeResult = _import_sibling("recurrence_evaluator").RecurrenceModeResult

_compat_api = _import_sibling("compat_api")
__all__ = _compat_api.PUBLIC_EXPORTS

from __future__ import annotations

import os
import copy
import hashlib
import importlib
import json
import time
from collections import OrderedDict
from functools import lru_cache, wraps
from types import MappingProxyType
from typing import Any, TypedDict

from nautical_core import cache_support, config_schema, config_support, warnings

_tomllib: Any = None
try:
    _tomllib = importlib.import_module("tomllib")  # Python 3.11+
except Exception:
    try:
        _tomllib = importlib.import_module("tomli")
    except Exception:
        _tomllib = None
tomllib: Any = _tomllib


_DEFAULTS = {
    "wrand_salt": config_schema.spec_default("wrand_salt"),
    "tz": config_schema.spec_default("tz"),
    "season_hemisphere": config_schema.spec_default("season_hemisphere"),
    "holiday_region": "",
    "anchor_file_dir": config_schema.spec_default("anchor_file_dir"),
    "omit_file_dir": config_schema.spec_default("omit_file_dir"),
    "anchor_presets": config_schema.spec_default("anchor_presets"),
    "omit_presets": config_schema.spec_default("omit_presets"),
    "business_calendar": config_schema.spec_default("business_calendar"),
    "astronomy": config_schema.spec_default("astronomy"),
}

_CONF_CACHE = None
_CONFIG_ERROR = ""
_CONFIG_ERROR_PATH = ""
_CONFIG_TASKDATA_OVERRIDE = ""
_CONFIG_LOADED = False
_CACHE_LOAD_MEM_MAX = 128
_CACHE_LOAD_MEM_TTL = 300
_CACHE_LOAD_MEM: OrderedDict[str, tuple[int, int, dict, float]] = OrderedDict()


class ConfigReloadResult(TypedDict, total=False):
    """Validated configuration reload outcome shared by lifecycle entry points."""

    ok: bool
    error: str
    source: str
    fingerprint: str
    scheduler_fingerprint: str


def env_flag_true(name: str, env_map: dict | None = None) -> bool:
    return config_support.env_flag_true(name, env_map=env_map)


def path_input_error(path_value: str) -> str | None:
    return config_support.path_input_error(path_value)


def normalized_abspath(path_value: str) -> str:
    return config_support.normalized_abspath(path_value)


def nearest_existing_dir(path_value: str) -> str | None:
    return config_support.nearest_existing_dir(path_value)


def world_writable_without_sticky(mode: int) -> bool:
    return config_support.world_writable_without_sticky(mode)


def path_safety_error(path_value: str, *, expect_dir: bool = True) -> str | None:
    return config_support.path_safety_error(path_value, expect_dir=expect_dir)


def validated_user_dir(
    path_value: str,
    *,
    label: str,
    trust_env: str = "",
    env_map: dict | None = None,
    warn_on_error: bool = True,
) -> str:
    return config_support.validated_user_dir(
        path_value,
        label=label,
        trust_env=trust_env,
        env_map=env_map,
        warn_on_error=warn_on_error,
    )


def _warn_env_config_missing(env_path: str) -> None:
    config_support.warn_env_config_missing(
        env_path,
        warn_once_per_day_any=warn_once_per_day_any,
    )


def _warn_missing_toml_parser(config_path: str) -> None:
    warnings.warn_missing_toml_parser(
        config_path,
        warn_once_per_day=warn_once_per_day,
        warn_once_per_day_any=warn_once_per_day_any,
    )


def _warn_toml_parse_error(config_path: str, err: Exception) -> None:
    warnings.warn_toml_parse_error(
        config_path,
        err,
        warn_once_per_day=warn_once_per_day,
        warn_once_per_day_any=warn_once_per_day_any,
    )


def _record_config_error(message: str, path: str = "") -> None:
    global _CONFIG_ERROR, _CONFIG_ERROR_PATH
    if not _CONFIG_ERROR:
        _CONFIG_ERROR = str(message or "configuration unavailable")
        _CONFIG_ERROR_PATH = str(path or "")


def _read_toml(path: str) -> dict:
    return _read_toml_result(path).data


def _read_toml_result(path: str):
    return config_support.read_toml_result(
        path,
        tomllib_mod=tomllib,
        warn_missing_toml_parser=_warn_missing_toml_parser,
        warn_toml_parse_error=_warn_toml_parse_error,
        error_sink=lambda message: _record_config_error(message, path),
    )


def _config_paths(taskdata: str | None = None) -> list[str]:
    selected_taskdata = taskdata if taskdata is not None else _CONFIG_TASKDATA_OVERRIDE
    return config_support.config_paths(
        warn_env_config_missing=_warn_env_config_missing,
        taskdata=selected_taskdata or None,
        error_sink=lambda message: _record_config_error(
            message,
            str(os.environ.get("NAUTICAL_CONFIG") or "").strip(),
        ),
    )


def _normalize_keys(d: dict) -> dict:
    return config_support.normalize_keys(d)


def _load_config(taskdata: str | None = None) -> dict:
    return config_support.load_config(
        defaults=_DEFAULTS,
        config_paths=lambda: _config_paths(taskdata),
        read_toml=_read_toml,
        read_toml_result=_read_toml_result,
        normalize_keys=_normalize_keys,
    )


def configuration_error() -> str:
    if not _CONFIG_ERROR:
        return ""
    configured = str(os.environ.get("NAUTICAL_CONFIG") or "").strip()
    if configured:
        active_path = os.path.abspath(os.path.expanduser(configured))
        return _CONFIG_ERROR if active_path == os.path.abspath(_CONFIG_ERROR_PATH) else ""
    try:
        active_paths = {os.path.abspath(path) for path in _config_paths()}
    except Exception:
        active_paths = set()
    return _CONFIG_ERROR if os.path.abspath(_CONFIG_ERROR_PATH) in active_paths else ""


def nautical_cache_dir() -> str:
    return cache_support.nautical_cache_dir(validated_user_dir=validated_user_dir)


def warn_once_per_day(key: str, message: str) -> None:
    warnings.warn_once_per_day(
        key,
        message,
        cache_dir=nautical_cache_dir(),
        require_diag=True,
    )


def warn_once_per_day_any(key: str, message: str) -> None:
    warnings.warn_once_per_day(
        key,
        message,
        cache_dir=nautical_cache_dir(),
        require_diag=False,
    )


def warn_rate_limited_any(key: str, message: str, min_interval_s: float = 3600.0) -> None:
    warnings.warn_rate_limited_any(
        key,
        message,
        cache_dir=nautical_cache_dir(),
        min_interval_s=min_interval_s,
    )


def _get_config() -> dict:
    global _CONF_CACHE
    out, _CONF_CACHE = config_support.get_config(_CONF_CACHE, load_config=_load_config)
    return out


_CONF = MappingProxyType(copy.deepcopy(_DEFAULTS))


def effective_config_snapshot() -> dict:
    """Return the effective immutable-at-call-time config and its source hint."""
    ensure_loaded()
    values = copy.deepcopy(dict(_CONF))
    source = "defaults"
    try:
        configured = str(os.environ.get("NAUTICAL_CONFIG") or "").strip()
        if configured:
            source = os.path.abspath(os.path.expanduser(configured))
        else:
            for path in _config_paths():
                if os.path.isfile(path):
                    source = os.path.abspath(path)
                    break
    except Exception:
        source = "auto"
    source_stat = None
    if source not in {"defaults", "auto"}:
        try:
            stat_result = os.stat(source)
            source_stat = (
                int(getattr(stat_result, "st_mtime_ns", int(stat_result.st_mtime * 1_000_000_000))),
                int(stat_result.st_size),
            )
        except Exception:
            source_stat = None
    fingerprint_payload = {"values": values, "source": source, "source_stat": source_stat}
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:24]
    return {"values": values, "source": source, "fingerprint": fingerprint}


_CONFIG_FINGERPRINT_CACHE: str | None = None
_CONFIG_FINGERPRINT_CACHE_KEY: str | None = None
_CONFIG_SOURCE_PATH_CACHE: str | None = None


def _config_source_hint() -> str:
    """Return the source identity without touching the filesystem."""
    try:
        configured = str(os.environ.get("NAUTICAL_CONFIG") or "").strip()
        if configured:
            return os.path.abspath(os.path.expanduser(configured))
        # Once the initial snapshot has selected a source, retain that source
        # identity. File edits are checked by configuration_drift(), not by
        # every hot cache-key call.
        return _CONFIG_SOURCE_PATH_CACHE or "auto"
    except Exception:
        return "auto"


def effective_config_fingerprint() -> str:
    """Return the cached config fingerprint for the current source identity.

    This is intentionally a hot-path lookup. Filesystem freshness belongs to
    ``configuration_drift()`` and explicit configuration reloads.
    """
    global _CONFIG_FINGERPRINT_CACHE, _CONFIG_FINGERPRINT_CACHE_KEY, _CONFIG_SOURCE_PATH_CACHE
    source_hint = _config_source_hint()
    if _CONFIG_FINGERPRINT_CACHE is None or source_hint != _CONFIG_FINGERPRINT_CACHE_KEY:
        snapshot = effective_config_snapshot()
        _CONFIG_FINGERPRINT_CACHE = str(snapshot.get("fingerprint") or "")
        _CONFIG_SOURCE_PATH_CACHE = str(snapshot.get("source") or "auto")
        _CONFIG_FINGERPRINT_CACHE_KEY = _CONFIG_SOURCE_PATH_CACHE
    return _CONFIG_FINGERPRINT_CACHE


_SCHEDULER_CONFIG_KEYS = (
    "wrand_salt",
    "tz",
    "season_hemisphere",
    "holiday_region",
    "anchor_file_dir",
    "omit_file_dir",
    "anchor_presets",
    "omit_presets",
    "business_calendar",
    "astronomy",
)

_SCHEDULER_FINGERPRINT_CACHE_KEY: str | None = None
_SCHEDULER_FINGERPRINT_CACHE: str | None = None


def scheduler_config_fingerprint() -> str:
    """Fingerprint only settings that can change recurrence projection results."""
    global _SCHEDULER_FINGERPRINT_CACHE_KEY, _SCHEDULER_FINGERPRINT_CACHE
    config_fingerprint = effective_config_fingerprint()
    if config_fingerprint == _SCHEDULER_FINGERPRINT_CACHE_KEY and _SCHEDULER_FINGERPRINT_CACHE is not None:
        return _SCHEDULER_FINGERPRINT_CACHE
    values = effective_config_snapshot().get("values")
    if not isinstance(values, dict):
        values = {}
    payload = {
        "version": 1,
        "values": {key: values.get(key) for key in _SCHEDULER_CONFIG_KEYS},
    }
    _SCHEDULER_FINGERPRINT_CACHE = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:24]
    _SCHEDULER_FINGERPRINT_CACHE_KEY = config_fingerprint
    return _SCHEDULER_FINGERPRINT_CACHE


_LOADED_CONFIG_FINGERPRINT = ""


def configuration_drift() -> dict:
    """Report whether the loaded process configuration differs from disk."""
    ensure_loaded()
    current = effective_config_snapshot()
    changed = current.get("fingerprint") != _LOADED_CONFIG_FINGERPRINT
    return {
        "changed": bool(changed),
        "status": "changed" if changed else "ok",
        "loaded_fingerprint": _LOADED_CONFIG_FINGERPRINT,
        "current_fingerprint": current.get("fingerprint", ""),
        "source": current.get("source", "unknown"),
    }


def ensure_loaded() -> None:
    """Load and validate configuration once, on first scheduling use."""
    global _CONFIG_LOADED, _CONF, _CONF_CACHE, _LOADED_CONFIG_FINGERPRINT
    if _CONFIG_LOADED:
        return
    loaded = _load_config()
    error = configuration_error()
    if error:
        _CONFIG_LOADED = True
        return
    _CONF_CACHE = copy.deepcopy(loaded)
    _CONF = MappingProxyType(copy.deepcopy(loaded))
    _refresh_config_exports()
    _CONFIG_LOADED = True
    _LOADED_CONFIG_FINGERPRINT = effective_config_fingerprint()


def conf_raw(key: str):
    return config_support.conf_raw(_CONF, key)


def conf_str(key: str, default: str) -> str:
    return config_support.conf_str(_CONF, key, default)


def conf_int(
    key: str,
    default: int,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    return config_support.conf_int(
        _CONF,
        key,
        default,
        min_value=min_value,
        max_value=max_value,
    )


def conf_bool(
    key: str,
    default: bool = False,
    true_values: set[str] | None = None,
    false_values: set[str] | None = None,
) -> bool:
    return config_support.conf_bool(
        _CONF,
        key,
        default=default,
        true_values=true_values,
        false_values=false_values,
    )


def conf_csv_or_list(key: str, default: list[str] | None = None, lower: bool = False) -> list[str]:
    return config_support.conf_csv_or_list(_CONF, key, default=default, lower=lower)


def conf_uda_field_list(key: str) -> list[str]:
    return config_support.conf_uda_field_list(_CONF, key)


def conf_schema_str(key: str) -> str:
    return conf_str(key, str(config_schema.spec_default(key)))


def conf_schema_int(key: str) -> int:
    spec = config_schema.CONFIG_SPECS[key]
    return conf_int(
        key,
        int(spec["default"]),
        min_value=spec.get("min"),
        max_value=spec.get("max"),
    )


def conf_schema_bool(
    key: str,
    *,
    true_values: set[str] | None = None,
    false_values: set[str] | None = None,
) -> bool:
    return conf_bool(
        key,
        bool(config_schema.spec_default(key)),
        true_values=true_values,
        false_values=false_values,
    )


def trueish(v, default=False):
    return config_support.trueish(v, default=default)


ANCHOR_YEAR_FMT = "MD"
WRAND_SALT = _CONF["wrand_salt"]
LOCAL_TZ_NAME = _CONF["tz"]
SEASON_HEMISPHERE = config_schema.normalized_choice("season_hemisphere", _CONF["season_hemisphere"])
HOLIDAY_REGION = _CONF["holiday_region"]
ANCHOR_FILE_DIR = _CONF["anchor_file_dir"]
OMIT_FILE_DIR = _CONF["omit_file_dir"]
ANCHOR_PRESETS = _CONF["anchor_presets"]
OMIT_PRESETS = _CONF["omit_presets"]
BUSINESS_CALENDAR_CONFIG = _CONF["business_calendar"]
ASTRONOMY_CONFIG = _CONF["astronomy"]

ENABLE_ANCHOR_CACHE = conf_schema_bool("enable_anchor_cache")
ENABLE_UDA_ALIASES = conf_schema_bool("enable_uda_aliases")
ANCHOR_CACHE_DIR_OVERRIDE = conf_schema_str("anchor_cache_dir")
ANCHOR_CACHE_TTL = conf_schema_int("anchor_cache_ttl")

CHAIN_COLOR_PER_CHAIN = conf_schema_bool(
    "chain_color_per_chain",
    true_values={"chain", "per-chain", "per"},
)
SHOW_TIMELINE_GAPS = conf_schema_bool(
    "show_timeline_gaps",
    false_values={"0", "no", "false", "off", "none"},
)
SHOW_ANALYTICS = conf_schema_bool(
    "show_analytics",
    false_values={"0", "no", "false", "off", "none"},
)
ANALYTICS_STYLE = conf_schema_str("analytics_style").lower()
ANALYTICS_STYLE = config_schema.normalized_choice("analytics_style", ANALYTICS_STYLE)
ANALYTICS_ONTIME_TOL_SECS = conf_schema_int("analytics_ontime_tol_secs")
DEBUG_WAIT_SCHED = conf_schema_bool(
    "debug_wait_sched",
    true_values={"1", "yes", "true", "on"},
)
CHECK_CHAIN_INTEGRITY = conf_schema_bool(
    "check_chain_integrity",
    true_values={"1", "yes", "true", "on"},
)
PANEL_MODE = config_schema.normalized_choice("panel_mode", conf_schema_str("panel_mode"))
LIVE_PANEL_DURATION_MS = conf_schema_int("live_panel_duration_ms")
LIVE_PANEL_FOOTER = conf_schema_str("live_panel_footer")
FAST_COLOR = conf_schema_bool("fast_color")
EXIT_PROGRESS = conf_schema_bool("exit_progress")
OUTBOX_DRAIN_MAX_ITEMS = conf_schema_int("outbox_drain_max_items")
MAX_CHAIN_WALK = conf_schema_int("max_chain_walk")
MAX_ANCHOR_ITER = conf_schema_int("max_anchor_iterations")
MAX_LINK_NUMBER = conf_schema_int("max_link_number")
SANITIZE_UDA = conf_schema_bool("sanitize_uda", true_values={"1", "yes", "true", "on"})
SANITIZE_UDA_MAX_LEN = conf_schema_int("sanitize_uda_max_len")
MAX_JSON_BYTES = conf_schema_int("max_json_bytes")
RECURRENCE_UPDATE_UDAS = tuple(conf_uda_field_list("recurrence_update_udas"))
CACHE_TTL_SECS = conf_schema_int("cache_ttl_secs")
CACHE_LOAD_MEM_MAX = conf_schema_int("cache_load_mem_max")
CACHE_LOAD_MEM_TTL = conf_schema_int("cache_load_mem_ttl")


def _refresh_config_exports() -> None:
    """Refresh module-level settings after loading a later Taskdata config."""
    globals().update(
        {
            "WRAND_SALT": _CONF["wrand_salt"],
            "LOCAL_TZ_NAME": _CONF["tz"],
            "SEASON_HEMISPHERE": config_schema.normalized_choice("season_hemisphere", _CONF["season_hemisphere"]),
            "HOLIDAY_REGION": _CONF["holiday_region"],
            "ANCHOR_FILE_DIR": _CONF["anchor_file_dir"],
            "OMIT_FILE_DIR": _CONF["omit_file_dir"],
            "ANCHOR_PRESETS": _CONF["anchor_presets"],
            "OMIT_PRESETS": _CONF["omit_presets"],
            "BUSINESS_CALENDAR_CONFIG": _CONF["business_calendar"],
            "ASTRONOMY_CONFIG": _CONF["astronomy"],
            "ENABLE_ANCHOR_CACHE": conf_schema_bool("enable_anchor_cache"),
            "ENABLE_UDA_ALIASES": conf_schema_bool("enable_uda_aliases"),
            "ANCHOR_CACHE_DIR_OVERRIDE": conf_schema_str("anchor_cache_dir"),
            "ANCHOR_CACHE_TTL": conf_schema_int("anchor_cache_ttl"),
            "CHAIN_COLOR_PER_CHAIN": conf_schema_bool(
                "chain_color_per_chain",
                true_values={"chain", "per-chain", "per"},
            ),
            "SHOW_TIMELINE_GAPS": conf_schema_bool(
                "show_timeline_gaps",
                false_values={"0", "no", "false", "off", "none"},
            ),
            "SHOW_ANALYTICS": conf_schema_bool(
                "show_analytics",
                false_values={"0", "no", "false", "off", "none"},
            ),
            "ANALYTICS_STYLE": config_schema.normalized_choice("analytics_style", conf_schema_str("analytics_style").lower()),
            "ANALYTICS_ONTIME_TOL_SECS": conf_schema_int("analytics_ontime_tol_secs"),
            "DEBUG_WAIT_SCHED": conf_schema_bool(
                "debug_wait_sched",
                true_values={"1", "yes", "true", "on"},
            ),
            "CHECK_CHAIN_INTEGRITY": conf_schema_bool(
                "check_chain_integrity",
                true_values={"1", "yes", "true", "on"},
            ),
            "PANEL_MODE": config_schema.normalized_choice("panel_mode", conf_schema_str("panel_mode")),
            "LIVE_PANEL_DURATION_MS": conf_schema_int("live_panel_duration_ms"),
            "LIVE_PANEL_FOOTER": conf_schema_str("live_panel_footer"),
            "FAST_COLOR": conf_schema_bool("fast_color"),
            "EXIT_PROGRESS": conf_schema_bool("exit_progress"),
            "OUTBOX_DRAIN_MAX_ITEMS": conf_schema_int("outbox_drain_max_items"),
            "MAX_CHAIN_WALK": conf_schema_int("max_chain_walk"),
            "MAX_ANCHOR_ITER": conf_schema_int("max_anchor_iterations"),
            "MAX_LINK_NUMBER": conf_schema_int("max_link_number"),
            "SANITIZE_UDA": conf_schema_bool("sanitize_uda", true_values={"1", "yes", "true", "on"}),
            "SANITIZE_UDA_MAX_LEN": conf_schema_int("sanitize_uda_max_len"),
            "MAX_JSON_BYTES": conf_schema_int("max_json_bytes"),
            "RECURRENCE_UPDATE_UDAS": tuple(conf_uda_field_list("recurrence_update_udas")),
            "CACHE_TTL_SECS": conf_schema_int("cache_ttl_secs"),
            "CACHE_LOAD_MEM_MAX": conf_schema_int("cache_load_mem_max"),
            "CACHE_LOAD_MEM_TTL": conf_schema_int("cache_load_mem_ttl"),
        }
    )


def reload_for_taskdata(taskdata: str | os.PathLike[str]) -> ConfigReloadResult:
    """Reload the selected config after resolving a Taskwarrior data directory.

    The normal import can happen before Taskwarrior exposes ``TASKDATA``. This
    refresh uses the same parser, precedence, and parse-error tracking as the
    normal loader, while allowing reconcile and doctor to supply the resolved
    directory without mutating the process environment.
    """
    global _CONF, _CONF_CACHE, _CONFIG_ERROR, _CONFIG_ERROR_PATH, _CONFIG_LOADED
    global _CONFIG_TASKDATA_OVERRIDE, _CONFIG_FINGERPRINT_CACHE
    global _CONFIG_FINGERPRINT_CACHE_KEY, _CONFIG_SOURCE_PATH_CACHE
    global _SCHEDULER_FINGERPRINT_CACHE_KEY, _SCHEDULER_FINGERPRINT_CACHE
    global _LOADED_CONFIG_FINGERPRINT

    explicit_config = bool(str(os.environ.get("NAUTICAL_CONFIG") or "").strip())
    resolved = ""
    if not explicit_config:
        resolved = os.path.abspath(os.path.expanduser(str(taskdata or "").strip()))
        if not resolved:
            return {"ok": False, "error": "Taskwarrior data directory is empty", "source": ""}
        _CONFIG_TASKDATA_OVERRIDE = resolved
    else:
        _CONFIG_TASKDATA_OVERRIDE = ""
    _CONFIG_ERROR = ""
    _CONFIG_ERROR_PATH = ""
    loaded = _load_config(taskdata=resolved or None)
    error = configuration_error()
    if error:
        return {"ok": False, "error": error, "source": resolved or "explicit"}

    _CONF_CACHE = copy.deepcopy(loaded)
    _CONF = MappingProxyType(copy.deepcopy(loaded))
    _CONFIG_LOADED = True
    _refresh_config_exports()
    _CONFIG_FINGERPRINT_CACHE = None
    _CONFIG_FINGERPRINT_CACHE_KEY = None
    _CONFIG_SOURCE_PATH_CACHE = None
    _SCHEDULER_FINGERPRINT_CACHE_KEY = None
    _SCHEDULER_FINGERPRINT_CACHE = None
    _LOADED_CONFIG_FINGERPRINT = effective_config_fingerprint()
    snapshot = effective_config_snapshot()
    scheduler_fingerprint = scheduler_config_fingerprint()
    return {
        "ok": True,
        "error": "",
        "source": str(snapshot.get("source") or resolved or "explicit"),
        "fingerprint": str(snapshot.get("fingerprint") or ""),
        "scheduler_fingerprint": scheduler_fingerprint,
    }


def ttl_lru_cache(maxsize: int = 128, ttl: float | None = None):
    ttl_val = CACHE_TTL_SECS if ttl is None else ttl

    def _decorator(fn):
        cached = lru_cache(maxsize=maxsize)(fn)
        last = {"t": time.time()}

        @wraps(fn)
        def _wrapper(*args, **kwargs):
            if ttl_val and (time.time() - last["t"] > ttl_val):
                cached.cache_clear()
                last["t"] = time.time()
            return cached(*args, **kwargs)

        setattr(_wrapper, "cache_clear", cached.cache_clear)
        setattr(_wrapper, "cache_info", cached.cache_info)
        return _wrapper

    return _decorator


# Explicit configuration is an opt-in contract and must be validated during
# import; automatic Taskdata discovery remains deferred until first use.
if str(os.environ.get("NAUTICAL_CONFIG") or "").strip():
    ensure_loaded()

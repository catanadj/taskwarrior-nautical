from __future__ import annotations

import os
import time
from collections import OrderedDict
from functools import lru_cache, wraps
from types import MappingProxyType

from nautical_core import cache_support, config_support, warnings

try:
    import tomllib  # Python 3.11+
except Exception:
    tomllib = None
if tomllib is None:
    try:
        import tomli

        tomllib = tomli
    except Exception:
        tomllib = None


_DEFAULTS = {
    "wrand_salt": "nautical|wrand|v3",
    "tz": "Europe/Bucharest",
    "holiday_region": "",
    "anchor_file_dir": "",
    "omit_file_dir": "",
}

_CONF_CACHE = None
_CACHE_LOAD_MEM_MAX = 128
_CACHE_LOAD_MEM_TTL = 300
_CACHE_LOAD_MEM: OrderedDict[str, tuple[int, int, dict, float]] = OrderedDict()


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


def _read_toml(path: str) -> dict:
    return config_support.read_toml(
        path,
        tomllib_mod=tomllib,
        warn_missing_toml_parser=_warn_missing_toml_parser,
        warn_toml_parse_error=_warn_toml_parse_error,
    )


def _config_paths() -> list[str]:
    return config_support.config_paths(warn_env_config_missing=_warn_env_config_missing)


def _normalize_keys(d: dict) -> dict:
    return config_support.normalize_keys(d)


def _load_config() -> dict:
    return config_support.load_config(
        defaults=_DEFAULTS,
        config_paths=_config_paths,
        read_toml=_read_toml,
        normalize_keys=_normalize_keys,
    )


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


_CONF = MappingProxyType(_get_config())


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


def trueish(v, default=False):
    return config_support.trueish(v, default=default)


ANCHOR_YEAR_FMT = "MD"
WRAND_SALT = _CONF["wrand_salt"]
LOCAL_TZ_NAME = _CONF["tz"]
HOLIDAY_REGION = _CONF["holiday_region"]
ANCHOR_FILE_DIR = _CONF["anchor_file_dir"]
OMIT_FILE_DIR = _CONF["omit_file_dir"]

ENABLE_ANCHOR_CACHE = conf_bool("enable_anchor_cache", False)
ANCHOR_CACHE_DIR_OVERRIDE = conf_str("anchor_cache_dir", "")
ANCHOR_CACHE_TTL = conf_int("anchor_cache_ttl", 0, min_value=0)

CHAIN_COLOR_PER_CHAIN = conf_bool(
    "chain_color_per_chain",
    False,
    true_values={"chain", "per-chain", "per"},
)
SHOW_TIMELINE_GAPS = conf_bool(
    "show_timeline_gaps",
    True,
    false_values={"0", "no", "false", "off", "none"},
)
SHOW_ANALYTICS = conf_bool(
    "show_analytics",
    True,
    false_values={"0", "no", "false", "off", "none"},
)
ANALYTICS_STYLE = conf_str("analytics_style", "clinical").lower()
if ANALYTICS_STYLE not in ("coach", "clinical"):
    ANALYTICS_STYLE = "clinical"
ANALYTICS_ONTIME_TOL_SECS = conf_int("analytics_ontime_tol_secs", 4 * 60 * 60, min_value=0)
VERIFY_IMPORT = conf_bool("verify_import", True)
DEBUG_WAIT_SCHED = conf_bool(
    "debug_wait_sched",
    False,
    true_values={"1", "yes", "true", "on"},
)
CHECK_CHAIN_INTEGRITY = conf_bool(
    "check_chain_integrity",
    False,
    true_values={"1", "yes", "true", "on"},
)
PANEL_MODE = conf_str("panel_mode", "rich").lower()
FAST_COLOR = conf_bool("fast_color", True)
EXIT_PROGRESS = conf_bool("exit_progress", True)
SPAWN_QUEUE_MAX_BYTES = conf_int("spawn_queue_max_bytes", 524288, min_value=0)
MAX_CHAIN_WALK = conf_int("max_chain_walk", 500, min_value=1)
MAX_ANCHOR_ITER = conf_int("max_anchor_iterations", 128, min_value=32, max_value=1024)
MAX_LINK_NUMBER = conf_int("max_link_number", 10000, min_value=1)
SANITIZE_UDA = conf_bool("sanitize_uda", False, true_values={"1", "yes", "true", "on"})
SANITIZE_UDA_MAX_LEN = conf_int("sanitize_uda_max_len", 1024, min_value=64, max_value=4096)
MAX_JSON_BYTES = conf_int("max_json_bytes", 10 * 1024 * 1024, min_value=1024, max_value=100 * 1024 * 1024)
RECURRENCE_UPDATE_UDAS = tuple(conf_uda_field_list("recurrence_update_udas"))
CACHE_TTL_SECS = conf_int("cache_ttl_secs", 3600, min_value=0)
CACHE_LOAD_MEM_MAX = conf_int("cache_load_mem_max", _CACHE_LOAD_MEM_MAX, min_value=16, max_value=4096)
CACHE_LOAD_MEM_TTL = conf_int("cache_load_mem_ttl", _CACHE_LOAD_MEM_TTL, min_value=0, max_value=86400)


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

        _wrapper.cache_clear = cached.cache_clear
        _wrapper.cache_info = cached.cache_info
        return _wrapper

    return _decorator

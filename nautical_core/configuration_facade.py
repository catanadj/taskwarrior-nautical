"""Compatibility boundary for facade-level configuration synchronization."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from os import PathLike
from typing import Any


def effective_snapshot(
    core_config: Any,
    refresh_exports: Callable[[], Any] | None = None,
) -> dict:
    """Refresh facade-visible exports before returning the canonical snapshot."""
    if callable(refresh_exports):
        refresh_exports()
    return core_config.effective_config_snapshot()


def scheduling_error(
    core_config: Any,
    config_error: str,
    timezone_error: str,
) -> str:
    """Combine canonical config and facade timezone errors."""
    core_config.ensure_loaded()
    canonical_error = core_config.configuration_error()
    if canonical_error:
        return canonical_error
    if config_error:
        return config_error
    return timezone_error


def reload_taskdata(
    core_config: Any,
    taskdata: str | PathLike[str],
    *,
    set_config_error: Callable[[str], Any],
    refresh_exports: Callable[[], Any],
) -> Mapping[str, Any]:
    """Reload canonical configuration and synchronize facade exports."""
    result = core_config.reload_for_taskdata(taskdata)
    if not result.get("ok"):
        error = str(result.get("error") or "configuration unavailable")
        set_config_error(f"Nautical configuration reload failed: {error}")
        raise RuntimeError(f"Nautical configuration reload failed: {error}")
    refresh_exports()
    return result


def validate_scheduling(
    *,
    core_config: Any,
    astronomy_config: Any,
    anchor_presets: Mapping[str, Any],
    omit_presets: Mapping[str, Any],
    resolve_anchor_presets: Callable[[str], str],
    validate_anchor_expr: Callable[[str], Any],
    resolve_omit_presets: Callable[[str], str],
    configured_business_calendars: Callable[[], Any],
    import_sibling: Callable[[str], Any],
) -> None:
    """Validate scheduling inputs without importing the facade module."""
    try:
        raw_season_mode = str(core_config._CONF.get("season_mode", "fixed") or "").strip().lower()
        valid_season_modes = core_config.config_schema.CONFIG_SPECS["season_mode"]["choices"]
        if raw_season_mode not in valid_season_modes:
            raise ValueError(
                f"season_mode must be 'fixed' or 'astronomical', got {raw_season_mode!r}"
            )
        import_sibling("astronomy").validate_configuration(astronomy_config)

        for name in sorted(dict(anchor_presets or {})):
            expression = resolve_anchor_presets(f"@{name}")
            validate_anchor_expr(expression)

        anchor_omit = import_sibling("anchor_omit")
        for name in sorted(dict(omit_presets or {})):
            anchor_omit.validate_omit_expr_strict(
                f"@{name}",
                validate_anchor_expr_cached=validate_anchor_expr,
                resolve_omit_presets=resolve_omit_presets,
            )

        clear_cache = getattr(configured_business_calendars, "cache_clear", None)
        if callable(clear_cache):
            clear_cache()
        configured_business_calendars()
    except Exception as exc:
        message = str(exc).strip() or type(exc).__name__
        raise RuntimeError(f"Invalid Nautical scheduling configuration: {message}") from exc


def sync_business_calendar_exports(namespace: dict[str, Any], calendar_module: Any) -> None:
    namespace["DEFAULT_BUSINESS_CALENDAR"] = calendar_module.DEFAULT_BUSINESS_CALENDAR
    namespace["business_calendar_displacement_for_date"] = calendar_module.business_calendar_displacement_for_date
    namespace["capture_business_calendar_displacements"] = calendar_module.capture_business_calendar_displacements


def configure_season_support(season_support: Any, hemisphere: str, mode: str, timezone_name: str) -> None:
    season_support.configure_hemisphere(hemisphere)
    season_support.configure_mode(mode)
    season_support.configure_timezone(timezone_name)


def sync_token_constants(namespace: dict[str, Any], tokenutil: Any) -> None:
    namespace["_MONTH_ALIAS"] = tokenutil.MONTH_ALIAS
    namespace["_WD_ABBR"] = tokenutil.WD_ABBR
    namespace["_WEEKLY_ALIAS"] = tokenutil.WEEKLY_ALIAS
    namespace["_MONTHLY_ALIAS"] = tokenutil.MONTHLY_ALIAS


def sync_exports(
    namespace: dict[str, Any],
    core_config: Any,
    names: tuple[str, ...],
    *,
    initial_sync: bool,
    season_override: bool,
) -> bool:
    if not initial_sync:
        return False
    value_for = getattr(core_config, "loaded_config_value", None)
    config_keys = {
        "WRAND_SALT": "wrand_salt",
        "LOCAL_TZ_NAME": "tz",
        "SEASON_HEMISPHERE": "season_hemisphere",
        "SEASON_MODE": "season_mode",
        "ANCHOR_FILE_DIR": "anchor_file_dir",
        "OMIT_FILE_DIR": "omit_file_dir",
        "ANCHOR_PRESETS": "anchor_presets",
        "OMIT_PRESETS": "omit_presets",
        "BUSINESS_CALENDAR_CONFIG": "business_calendar",
        "ASTRONOMY_CONFIG": "astronomy",
    }
    for name in names:
        config_name = name if not name.startswith("_") else name[1:]
        if name != "SEASON_HEMISPHERE" or not season_override:
            fallback = getattr(core_config, config_name)
            key = config_keys.get(config_name, config_name.lower())
            namespace[name] = (
                value_for(key, fallback)
                if callable(value_for)
                else fallback
            )
    return True


def refresh_loaded_state(
    namespace: dict[str, Any],
    core_config: Any,
    names: tuple[str, ...],
    *,
    facade_synced: bool,
    season_support: Any,
    refresh_timezone: Callable[[], Any],
    configure_season_support: Callable[[], Any],
    conf_int: Callable[..., int],
) -> bool:
    initial_sync = not facade_synced
    value_for = getattr(core_config, "loaded_config_value", None)
    configured_hemisphere = (
        value_for("season_hemisphere", "north")
        if callable(value_for)
        else getattr(core_config, "SEASON_HEMISPHERE", "north")
    )
    season_override = (
        not facade_synced
        and season_support.active_hemisphere() != configured_hemisphere
    )
    synced = sync_exports(
        namespace,
        core_config,
        names,
        initial_sync=initial_sync,
        season_override=season_override,
    ) or facade_synced
    namespace["_CONF"] = core_config._CONF
    namespace["CONFIG_ERROR"] = core_config.configuration_error()
    if namespace.get("MAX_ANCHOR_DNF_TERMS") == namespace.get("_CONFIG_DEFAULT_MAX_ANCHOR_DNF_TERMS", 10_000):
        namespace["MAX_ANCHOR_DNF_TERMS"] = conf_int(
            "max_anchor_dnf_terms", 10_000, min_value=64, max_value=200_000
        )
    refresh_timezone()
    if initial_sync and not season_override:
        configure_season_support()
    else:
        season_support.configure_mode(namespace["SEASON_MODE"])
        season_support.configure_timezone(namespace["LOCAL_TZ_NAME"])
    return synced


__all__ = (
    "effective_snapshot", "scheduling_error", "reload_taskdata", "validate_scheduling",
    "sync_business_calendar_exports", "configure_season_support", "sync_token_constants", "sync_exports", "refresh_loaded_state",
)

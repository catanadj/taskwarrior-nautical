"""Public cache entry points bound to one core facade instance."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def for_core(module: Any, *, namespace: dict[str, Any] | None = None):
    """Create cache APIs without sharing cache state across core loaders."""
    core = namespace if namespace is not None else vars(module)
    cache_dir_state: list[str | None] = [None]
    import_sibling = core.get("_import_sibling", module._import_sibling)
    cache_support = import_sibling("cache_support")
    cache_locking = import_sibling("cache_locking")

    def cache_dir() -> str:
        current = core.get("_CACHE_DIR", cache_dir_state[0])
        chosen = cache_locking.cache_dir(
            current,
            anchor_cache_dir_override=core["ANCHOR_CACHE_DIR_OVERRIDE"],
            nautical_cache_dir_path=core["_nautical_cache_dir"](),
            validated_user_dir=core["_validated_user_dir"],
            select_cache_dir=cache_support.select_cache_dir,
        )
        cache_dir_state[0] = chosen
        core["_CACHE_DIR"] = chosen
        return chosen

    def cache_key(
        acf: str,
        anchor_mode: str,
        *,
        business_calendar_fingerprint: str = "",
    ) -> str:
        config_fingerprint = core["scheduler_config_fingerprint"]()
        profile_fingerprint = (
            f"{business_calendar_fingerprint}|season:{core['SEASON_HEMISPHERE']}"
            f"|config:{config_fingerprint}|parser:2|cache:2"
        )
        return cache_support.cache_key(
            acf,
            anchor_mode,
            business_calendar_fingerprint=profile_fingerprint,
            anchor_year_fmt=core["ANCHOR_YEAR_FMT"],
            wrand_salt=core["WRAND_SALT"],
            local_tz_name=core["LOCAL_TZ_NAME"],
            holiday_region=core["HOLIDAY_REGION"],
        )

    def cache_path(key: str) -> str:
        return cache_support.cache_path(cache_dir(), key)

    def cache_lock_path(key: str) -> str:
        return cache_support.cache_lock_path(cache_dir(), key)

    return SimpleNamespace(
        _cache_dir=cache_dir,
        _cache_key=cache_key,
        _cache_path=cache_path,
        _cache_lock_path=cache_lock_path,
        cache_load=module._cache_load_impl,
        cache_save=module._cache_save_impl,
        cache_gc=module._cache_gc_impl,
        cache_key_for_task=module._cache_key_for_task_impl,
    )


__all__ = ("for_core",)

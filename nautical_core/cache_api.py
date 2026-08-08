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

    def quarantine_cache(key: str, path: str) -> bool:
        """Move a broken cache entry aside so future reads become clean misses."""
        try:
            with core["_cache_lock"](key) as locked:
                if not locked or not core["os"].path.exists(path):
                    return False
                target = f"{path}.bad.{core['os'].getpid()}.{core['time'].time_ns()}"
                core["os"].replace(path, target)
                core["_CACHE_LOAD_MEM"].pop(key, None)
                return True
        except Exception:
            return False

    def cache_load_impl(key: str) -> dict | None:
        return core["_cache_payload"].cache_load(
            key,
            enable_anchor_cache=core["ENABLE_ANCHOR_CACHE"],
            cache_path=cache_path,
            anchor_cache_ttl=core["ANCHOR_CACHE_TTL"],
            time_mod=core["time"],
            cache_load_mem=core["_CACHE_LOAD_MEM"],
            cache_load_mem_ttl=core["_CACHE_LOAD_MEM_TTL"],
            clone_cache_payload=core["_clone_cache_payload"],
            normalize_dnf_cached=core["_normalize_dnf_cached"],
            cache_payload_shape_ok=core["_cache_payload_shape_ok"],
            cache_load_mem_max=core["_CACHE_LOAD_MEM_MAX"],
            diag=core["diag"],
            quarantine_cache=quarantine_cache,
            os_mod=core["os"],
            json_mod=core["json"],
            zlib_mod=core["zlib"],
            base64_mod=core["base64"],
        )

    def cache_save_impl(key: str, obj: dict) -> bool:
        return core["_cache_payload"].cache_save(
            key,
            obj,
            enable_anchor_cache=core["ENABLE_ANCHOR_CACHE"],
            json_mod=core["json"],
            zlib_mod=core["zlib"],
            base64_mod=core["base64"],
            cache_path=cache_path,
            cache_dir=cache_dir,
            cache_lock=core["_cache_lock"],
            diag=core["diag"],
            os_mod=core["os"],
            tempfile_mod=core["tempfile"],
            cache_atomic_replace=core["_cache_atomic_replace"],
            cache_load_mem=core["_CACHE_LOAD_MEM"],
        )

    def cache_gc_impl(
        *,
        max_entries: int = 512,
        stale_tmp_age: float = 86400.0,
        stale_lock_age: float = 86400.0,
    ) -> dict:
        """Prune expired and orphaned anchor cache files outside hook hot paths."""
        return core["_cache_payload"].cache_gc(
            cache_dir(),
            ttl=core["ANCHOR_CACHE_TTL"],
            max_entries=max_entries,
            stale_tmp_age=stale_tmp_age,
            stale_lock_age=stale_lock_age,
            cache_lock=core["_cache_lock"],
            stale_lock_check=lambda path, age: core["_safe_lock_stale_pid"](path, age)
            and (core["_safe_lock_age"](path) or 0.0) >= float(age),
            time_mod=core["time"],
            os_mod=core["os"],
        )

    ttl_lru_cache = core["_ttl_lru_cache"]

    @ttl_lru_cache(maxsize=1024)
    def cache_key_for_task_cached(
        anchor_expr: str,
        anchor_mode: str,
        fmt: str,
        business_calendar_fingerprint: str = "",
        config_fingerprint: str = "",
    ) -> str:
        _ = config_fingerprint
        return core["_cache_payload"].cache_key_for_task_cached(
            anchor_expr,
            anchor_mode,
            fmt,
            business_calendar_fingerprint,
            build_acf=core["build_acf"],
            cache_key=cache_key,
        )

    def cache_key_for_task_impl(
        anchor_expr: str,
        anchor_mode: str,
        calendar_fingerprint: str | None = None,
    ) -> str:
        if calendar_fingerprint is None:
            calendar_fingerprint = core["business_calendar_fingerprint"]()
        return cache_key_for_task_cached(
            anchor_expr or "",
            anchor_mode or "",
            core["_yearfmt"](),
            calendar_fingerprint,
            core["effective_config_fingerprint"](),
        )

    return SimpleNamespace(
        _cache_dir=cache_dir,
        _cache_key=cache_key,
        _cache_path=cache_path,
        _cache_lock_path=cache_lock_path,
        _quarantine_cache=quarantine_cache,
        _cache_load_impl=cache_load_impl,
        _cache_save_impl=cache_save_impl,
        _cache_gc_impl=cache_gc_impl,
        _cache_key_for_task_cached=cache_key_for_task_cached,
        _cache_key_for_task_impl=cache_key_for_task_impl,
        cache_load=cache_load_impl,
        cache_save=cache_save_impl,
        cache_gc=cache_gc_impl,
        cache_key_for_task=cache_key_for_task_impl,
    )


__all__ = ("for_core",)
